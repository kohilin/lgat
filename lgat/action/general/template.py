import glob
import json
import logging
import os

from collections import OrderedDict
from more_itertools import flatten

from nltk.corpus import wordnet as wn

from ..base import SlotBase, TemplateBase, TemplateCollectionBase
from ..masker import MaskerBase, PoSMasker, RoleMasker, StopWordsMasker, LastObservationMasker, LMMasker, EnforceMasker, AndOrMaskAggregator
from ... import helper
from ...vocab import VocabDict


ACTION_DEF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'definitions')
ACTION_DEFINITIONS = {'v1': os.path.join(ACTION_DEF_DIR, 'v1')}

HYPONYM_CLOSURE_CACHE = {}
HYPONYM_CLOSURE_CACHE_FILE = os.path.join(helper.RESOURCE_PATH, 'lgat_v1_hyponym_closure.cache')
FOUND_NEW_HYPONYM = False


logger = logging.getLogger(__name__)


def dump_hyponym_closure_cache():
    string_dict = {}
    for k, v in HYPONYM_CLOSURE_CACHE.items():
        string_dict[k.name()] = [x.name() for x in v]

    with open(HYPONYM_CLOSURE_CACHE_FILE, 'w') as f:
        json.dump(string_dict, f)


def load_hyponym_closure_cache():
    logger.info(f"Loading hyponyms from cache {HYPONYM_CLOSURE_CACHE_FILE} ... ")
    with open(HYPONYM_CLOSURE_CACHE_FILE, 'r') as f:
        d = json.load(f)

    instance_dict = {}

    for k, v in d.items():
        instance_dict[wn.synset(k)] = [wn.synset(w) for w in v]
    logger.info("Done !!")
    return instance_dict


def lookup_hyponyms(wn_nodes, is_predicate):
    global HYPONYM_CLOSURE_CACHE
    if len(HYPONYM_CLOSURE_CACHE) != 0:
        pass
    elif os.path.exists(HYPONYM_CLOSURE_CACHE_FILE):
        HYPONYM_CLOSURE_CACHE = load_hyponym_closure_cache()
    else:
        logger.info(f"Lookup hyponyms of {wn_nodes} from WordNet ... (This will take a bit time, use cache from the second run.)")

    synsets, words = [], []
    if is_predicate:
        for node_id in wn_nodes:
            synset = wn.synset(node_id)
            synsets.append(synset)
            # TODO: use .lemma_names()
            lemma = synset.lemmas()[0].name()
            words.append(lemma)
        return synsets, words
    else:
        for node_id in wn_nodes:
            synset = wn.synset(node_id)
            if synset not in HYPONYM_CLOSURE_CACHE:
                hyponynms = \
                    list(synset.closure(lambda x: x.hyponyms())) + [synset]
                HYPONYM_CLOSURE_CACHE[synset] = hyponynms
                global FOUND_NEW_HYPONYM
                FOUND_NEW_HYPONYM = True
            else:
                hyponynms = HYPONYM_CLOSURE_CACHE[synset]
            hyponym_lemmas = [h.lemma_names() for h in hyponynms]
            hyponym_lemmas = [l.lower() for l in flatten(hyponym_lemmas) if
                              '_' not in l]

            words += hyponym_lemmas
            synsets += hyponynms

        return synsets, words


class GeneralSlot(SlotBase):
    def __init__(self, fn_frame=None, vn_role=None, vn_frame=None, fn_roles=None,
                 wn_nodes=None, pos='n', words=None, enforce_words=None,
                 require_observed=True, lm_suffix=None, is_nullable=False,
                 is_special_role=False, **kwargs):
        props = dict()

        props['fn_frame'] = fn_frame
        props['vn_frame'] = vn_frame

        if is_special_role:
            props['vn_role'] = None
        else:
            props['vn_role'] = vn_role
        props['fn_roles'] = fn_roles

        props['wn_nodes'] = wn_nodes
        props['wn_root_nodes'] = wn_nodes
        if wn_nodes:
            is_predicate = vn_role == 'Predicate'
            hyponym_synsets, hyponym_words = lookup_hyponyms(wn_nodes, is_predicate=is_predicate)
            props['lexicon'] = hyponym_words
            props['wn_nodes'] = hyponym_synsets
        else:
            if vn_role == 'Preposition':
                props['lexicon'] = words
            elif vn_role == 'Null':
                props['lexicon'] = []

        props['enforce_words'] = enforce_words or []
        props['is_null'] = vn_role == 'Null'
        props['is_nullable'] = props['is_null'] or is_nullable
        props['require_observed'] = require_observed
        props['lm_suffix'] = lm_suffix
        props['pos'] = pos
        props['name'] = vn_role

        super(GeneralSlot, self).__init__(**props)

    def __str__(self):
        return f'{self.template}({self.name})'


class GeneralTemplate(TemplateBase):
    SLOT_NAMES = None

    def __init__(self, slots, name, **kwargs):
        slots_dict = OrderedDict()
        for sn, s in zip(GeneralTemplate.SLOT_NAMES, slots):
            slots_dict[sn] = s

        super(GeneralTemplate, self).__init__(slots_dict, name=name, **kwargs)

        for s in self.slots.values():
            s.set_property(template=self)

    @classmethod
    def set_slot_names(cls, slot_names):
        GeneralTemplate.SLOT_NAMES = slot_names

    @classmethod
    def from_yaml(cls, dir):
        meta_yaml_file = glob.glob(os.path.join(dir, '_meta.yaml'))
        meta = helper.load_yaml(meta_yaml_file[0])

        logger.info(f"Initialize LGAT with {meta['name']} templates")

        GeneralTemplate.set_slot_names(meta['slot_names'])

        action_yaml_files = [f for f in glob.glob(os.path.join(dir, '*'))]
        templates = []
        for ayf in action_yaml_files:
            action = helper.load_yaml(ayf)
            filename = os.path.split(ayf)[-1]

            if filename.startswith('_'):
                continue

            fn_frame = action['fn_frame']
            vn_frame = action['vn_frame']
            slots = []
            for s_name in meta['slot_names']:
                s_args = action['slots'][s_name]
                s = GeneralSlot(fn_frame=fn_frame, vn_frame=vn_frame, **s_args)
                slots.append(s)

            templates.append(GeneralTemplate(slots, action['name']))

        return templates


class GeneralTemplateCollection(TemplateCollectionBase):
    def __init__(self, templates='v1', vocab=None, masker=None):
        if isinstance(templates, str):
            if templates in ACTION_DEFINITIONS:
                path = ACTION_DEFINITIONS[templates]
            else:
                path = templates
            templates = GeneralTemplate.from_yaml(path)

        if vocab is None:
            vocab = VocabDict()

        if masker is None:
            masker = self.build_masker(vocab, ['pos', 'role', 'stopwords', 'observation_last', 'lm'])

        slot_names = GeneralTemplate.SLOT_NAMES

        if not os.path.exists(HYPONYM_CLOSURE_CACHE_FILE) or FOUND_NEW_HYPONYM:
            dump_hyponym_closure_cache()

        super(GeneralTemplateCollection, self).__init__(templates, vocab, masker, slot_names)

    @classmethod
    def build_masker(cls, vocab, masker_names):
        MaskerBase.set_vocab(vocab)
        maskers = list()

        if 'pos' in masker_names:
            maskers.append(PoSMasker())
        if 'role' in masker_names:
            maskers.append(RoleMasker())
        if 'stopwords' in masker_names:
            maskers.append(StopWordsMasker())
        if 'observation_last' in masker_names:
            maskers.append(LastObservationMasker())
        if 'lm' in masker_names:
            maskers.append(LMMasker())

        return AndOrMaskAggregator(maskers, [EnforceMasker()])
