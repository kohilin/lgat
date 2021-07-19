import re

import torch

from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..helper import DEVICE, DIRECTION_WORDS
from .. import resources


def sanity_check_mask(mask):
    if torch.sum(mask) == 0:
        raise ValueError(f"A zero mask (i.e., all elements are zero) was found. At least one element should be 1.")


def proc_generate_mask_slot(func):
    def wrapper(*args, **kwargs):
        masker_cls = args[0]
        slot = args[1]
        masker_cls.check_masker_attribute(slot, **kwargs)

        mask = func(*args, **kwargs)
        if slot.is_nullable:
            mask[0] = 1

        return mask

    return wrapper


class MaskerBase(object):
    VOCAB = None
    REQUIRED_SLOT_ATTRS = []
    REQUIRED_KWARGS = []

    def generate_mask(self, template, target_slot=None, **kwargs):
        masks = OrderedDict()

        for name, slot in template.slots.items():
            if target_slot and name not in target_slot:
                continue

            mask = self.generate_mask_slot(slot, **kwargs)

            masks[name] = mask

        return masks

    def generate_mask_batch(self, templates, target_slot=None, **kwargs):
        masks = OrderedDict()

        slot_names = templates[0].slots.keys()

        for name in slot_names:
            if target_slot and name not in target_slot:
                continue

            slots = [t.slots[name] for t in templates]

            masks[name] = self.generate_mask_slot_batch(slots, **kwargs)

        return masks

    def generate_mask_slot(self, slot, **kwargs):
        raise NotImplementedError

    def generate_mask_slot_batch(self, slots, **kwargs):
        masks = []

        for slot in slots:
            masks.append(self.generate_mask_slot(slot, **kwargs))

        return torch.stack(masks)

    @classmethod
    def set_vocab(cls, vocab):
        MaskerBase.VOCAB = vocab

    @classmethod
    def init_mask(cls, n=None, fill=0):
        if n is None:
            n = cls.VOCAB.size

        if fill == 0:
            mask = torch.zeros(n, device=DEVICE).long()
        elif fill == 1:
            mask = torch.ones(n, device=DEVICE).long()
        else:
            assert False, "fill must be 0 or 1"

        return mask

    @classmethod
    def null_mask(cls):
        mask = cls.init_mask()
        mask[cls.VOCAB.null_id] = 1
        return mask

    @classmethod
    def generate_mask_from_list(cls, lst):
        vec = cls.init_mask()
        for w in lst:
            if cls.VOCAB.has(w):
                widx = cls.VOCAB.to_id(w)
                vec[widx] = 1
        return vec

    @classmethod
    def check_masker_attribute(cls, slot, **kwargs):
        if slot is not None:
            for sa in cls.REQUIRED_SLOT_ATTRS:
                if sa not in slot:
                    raise ValueError(f"Required attribute is missing in {slot}, {cls.__class__.__name__} requires '{sa}' attribute.")
        if kwargs is not None:
            for ka in cls.REQUIRED_KWARGS:
                if ka not in kwargs:
                    raise ValueError(f"Required kwargs is missing. {cls.__class__.__name__} requires {ka} is given as kwargs.")

    @classmethod
    def check_mask(cls, mask):
        non_zeros = [idx for (idx, m) in enumerate(mask) if m != 0]
        return MaskerBase.VOCAB.to_str_list(non_zeros)

    @classmethod
    def vocab_size(cls):
        assert cls.VOCAB is not None, "VOCAB has not been initialzied."
        return cls.VOCAB.size()


class DummyMasker(MaskerBase):
    def generate_mask_slot(self, slot, **kwargs):
        return self.init_mask(fill=1)


class ListMaskerBase(MaskerBase):
    KEY = None

    @proc_generate_mask_slot
    def generate_mask_slot(self, slot, **kwargs):
        assert self.KEY is not None

        if self.KEY in slot:
            possible_words = slot[self.KEY]
        elif self.KEY in kwargs:
            possible_words = kwargs[self.KEY]
        else:
            assert False

        return self.generate_mask_from_list(possible_words)

    @classmethod
    def check_masker_attribute(cls, slot, **kwargs):
        if cls.KEY not in slot and cls.KEY not in kwargs:
            raise ValueError(f"A list named with '{cls.KEY}' is missing, which must be defined as a slot attribute or given with kwargs for {cls.__class__.__name__}.")


class EnforceMasker(ListMaskerBase):
    KEY = 'enforce_words'


class RoleMasker(ListMaskerBase):
    KEY = 'lexicon'


class PoSMasker(MaskerBase):
    REQUIRED_SLOT_ATTRS = ['pos']

    def __init__(self, pos_mappings='default', verbs=None, nouns=None, preps=None):
        super(PoSMasker, self).__init__()
        if pos_mappings == 'default':
            self.pos_mappings = {
                'v': resources.load_wordnet_word_list('v'),
                'n': resources.load_wordnet_word_list('n'),
                'p': resources.load_prepositions()
            }
        else:
            self.pos_mappings = pos_mappings

        if verbs:
            self.pos_mappings['v'].update(verbs)
        if nouns:
            self.pos_mappings['n'].update(nouns)
        if preps:
            self.pos_mappings['p'].update(preps)

        self.pos_mappings['n'].update(DIRECTION_WORDS)

        self.pos_masks = dict()
        for k, v in self.pos_mappings.items():
            self.pos_masks[k] = self.generate_mask_from_list(v)

    @proc_generate_mask_slot
    def generate_mask_slot(self, slot, **kwargs):
        pos = slot['pos']
        if pos is None:
            return self.null_mask()
        else:
            return self.pos_masks[slot['pos']]


class ObservationMasker(MaskerBase):
    REQUIRED_SLOT_ATTRS = ['require_observed']
    REQUIRED_KWARGS = ['observed']

    @proc_generate_mask_slot
    def generate_mask_slot(self, slot, **kwargs):
        observed = kwargs['observed']

        if slot['require_observed']:
            return self.generate_mask_from_list(observed)
        else:
            return self.init_mask(fill=1)


class LastObservationMasker(MaskerBase):
    REQUIRED_SLOT_ATTRS = ['require_observed']
    REQUIRED_KWARGS = ['observed_last']

    @proc_generate_mask_slot
    def generate_mask_slot(self, slot, **kwargs):
        observed_last = kwargs['observed_last']
        if slot['require_observed']:
            return self.generate_mask_from_list(observed_last)
        else:
            return self.init_mask(fill=1)

    def generate_mask_slot_batch(self, slots, **kwargs):
        masks = []

        observed_last = kwargs['observed_last']

        if len(slots) != len(observed_last):
            observed_last = [observed_last[0]] * len(slots)

        for s, ol in zip(slots, observed_last):
            masks.append(self.generate_mask_slot(s, observed_last=ol))

        return torch.stack(masks)


class StopWordsMasker(MaskerBase):
    def __init__(self):
        stopwords = resources.load_stopwords()
        self.__mask = 1 - self.generate_mask_from_list(stopwords)
        self.__mask[self.VOCAB.null_id] = 1  # Null

    @proc_generate_mask_slot
    def generate_mask_slot(self, slot, **kwargs):
        return self.__mask


class LMMasker(MaskerBase):
    REQUIRED_SLOT_ATTRS = ['lm_suffix']
    REQUIRED_KWARGS = ['obs']
    MODEL = None
    TOKENIZER = None

    def __init__(self, model_name='gpt2', k=50):
        assert model_name == 'gpt2', 'only gpt2 is supported for now.'
        self.model_name = model_name
        self.k = k
        self.vocab = None
        self.__cache = {}

    @proc_generate_mask_slot
    def generate_mask_slot(self, slot, **kwargs):
        kwargs['obs'] = [kwargs['obs']]
        return self.generate_mask_slot_batch([slot], **kwargs)[0, :]

    def generate_mask_slot_batch(self, slots, **kwargs):
        if LMMasker.MODEL is None:
            self.__init_model()

        if 'suffix' in kwargs:
            suffix = [kwargs['suffix']] * len(slots)
        else:
            suffix = [s['lm_suffix'] for s in slots]

        batch_size = len(slots)
        tar_obs = kwargs['obs']

        if batch_size != len(tar_obs):  # used when generate_mask_all=True
            tar_obs = [tar_obs[0]] * batch_size

        masks, new_obs, new_obs_idx = [None] * batch_size, [], []
        for idx, (o, s) in enumerate(zip(tar_obs, suffix)):
            if not s:
                masks[idx] = self.init_mask(fill=1)
                continue
            o_w_s = o + ' ' + s

            if o_w_s in self.__cache:
                masks[idx] = self.__cache[o_w_s]
            else:
                new_obs.append(o_w_s)
                new_obs_idx.append(idx)

        if new_obs:
            new_masks = self._generate_mask(new_obs)

            for idx, o, m in zip(new_obs_idx, new_obs, new_masks):
                self.__cache[o] = m
                masks[idx] = m

        masks = torch.stack(masks)

        return masks

    def _get_last_logits(self, texts):
        inputs = self.TOKENIZER(
            texts, add_special_tokens=False, padding=True,
            return_tensors='pt', truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.MODEL(**inputs)

        last_logits = []
        for l, a in zip(outputs.logits, inputs['attention_mask']):
            last_logits.append(l[a != 0][-1, :])

        last_logits = torch.stack(last_logits)

        return last_logits

    def _get_topk_tokens(self, logits):
        _, indices = torch.topk(logits, k=self.k)

        tokens = []
        for l in indices.tolist():
            tmp = self.TOKENIZER.convert_ids_to_tokens(l)
            tokens.append([re.sub(r'Ġ', '', t).lower() for t in tmp])

        return tokens

    def _generate_unknown_word_mask(self, obs):
        unknown_words_mask = []

        for o in obs:
            tokenized_o = set(word_tokenize(o.lower()))

            unknown_words = tokenized_o.difference(self.vocab)

            tmp = []
            for w in unknown_words:
                x = wn.morphy(w)
                if x is not None and x not in unknown_words:
                    tmp.append(x)
            unknown_words.update(tmp)

            unknown_words_mask.append(
                self.generate_mask_from_list(unknown_words))

        return unknown_words_mask

    def _generate_topk_tokens(self, obs):
        logits = self._get_last_logits(obs)
        tokens = self._get_topk_tokens(logits)
        return tokens

    def _generate_mask(self, obs):
        tokens = self._generate_topk_tokens(obs)
        masks = [self.generate_mask_from_list(t) for t in tokens]

        unknown_words_masks = self._generate_unknown_word_mask(obs)

        return [m | um for (m, um) in zip(masks, unknown_words_masks)]

    def _reset_cache(self):
        self.__cache = {}

    def __init_model(self):
        LMMasker.MODEL = AutoModelForCausalLM.from_pretrained(self.model_name)
        LMMasker.MODEL.eval()
        LMMasker.MODEL.to(DEVICE)

        LMMasker.TOKENIZER = AutoTokenizer.from_pretrained(self.model_name)
        LMMasker.TOKENIZER.pad_token = LMMasker.TOKENIZER.eos_token

        self.vocab = set([w.replace('Ġ', '') for w in self.TOKENIZER.vocab])


class MaskAggregatorBase(MaskerBase):
    def __init__(self, maskers):
        self.maskers = maskers

    def aggregate_masks(self, masks_per_slot, **kwargs):

        aggr_fn = kwargs.get('fn', self.aggregate_mask_slot)

        aggregated_masks = OrderedDict()

        for slot_name, masks in masks_per_slot.items():
            aggregated_masks[slot_name] = aggr_fn(masks, **kwargs)

        return aggregated_masks

    def generate_mask(self, template, **kwargs):
        masks = [m.generate_mask(template, **kwargs) for m in self.maskers]

        masks_per_slot = self.convert_nest_of_masker_results(masks)

        aggr_mask = self.aggregate_masks(masks_per_slot, **kwargs)

        if kwargs.get('return_all', False):
            return aggr_mask, masks_per_slot
        else:
            return aggr_mask

    def generate_mask_batch(self, templates, **kwargs):
        masks = [m.generate_mask_batch(templates, **kwargs) for m in
                 self.maskers]

        masks_per_slot = self.convert_nest_of_masker_results(masks)

        return self.aggregate_masks(masks_per_slot)

    @classmethod
    def aggregate_mask_slot(cls, masks, **kwargs):
        raise NotImplementedError

    @classmethod
    def convert_nest_of_masker_results(cls, masks):
        masks_for_each_slot = {}

        slot_names = masks[0].keys()

        for slot_name in slot_names:
            tar_masks = []
            for m in masks:
                tar_masks.append(m[slot_name])

            masks_for_each_slot[slot_name] = tar_masks

        return masks_for_each_slot


class AndMaskAggregator(MaskAggregatorBase):
    @classmethod
    def aggregate_mask_slot(cls, masks, **kwargs):
        and_mask = masks[0]

        for m in masks[1:]:
            and_mask = and_mask & m

        return and_mask

    def check(self, t, slot, target, **kwargs):
        _, masks = self.generate_mask(t, return_all=True, **kwargs)
        slot_masks = masks[slot]
        if kwargs.pop('check_mask', False):
            return {m: sm[target] == 1 for (m, sm) in zip(self.maskers, slot_masks)}
        else:
            return {m: (sm[target] == 1, MaskerBase.check_mask(sm)) for (m, sm) in zip(self.maskers, slot_masks)}


class OrMaskAggregator(MaskAggregatorBase):

    @classmethod
    def aggregate_mask_slot(cls, masks, **kwargs):
        mask = masks[0]

        for m in masks[1:]:
            mask = mask | m

        return mask


class AndOrMaskAggregator(MaskAggregatorBase):
    def __init__(self, and_maskers, or_maskers):
        self.and_aggregator = AndMaskAggregator(and_maskers)
        self.or_aggregator = OrMaskAggregator(or_maskers)

    def __generate_mask(self, and_mask, or_mask):
        masks_per_slot = OrderedDict()
        for slot_name in and_mask.keys():
            masks_per_slot[slot_name] = [and_mask[slot_name], or_mask[slot_name]]

        masks = self.aggregate_masks(masks_per_slot, fn=OrMaskAggregator.aggregate_mask_slot)
        return masks

    def generate_mask(self, template, **kwargs):
        and_mask = self.and_aggregator.generate_mask(template, **kwargs)
        or_mask = self.or_aggregator.generate_mask(template, **kwargs)
        if kwargs.get('return_all', False):
            and_aggr_mask = and_mask[0]
            and_mask_per_slot = and_mask[1]
            or_aggr_mask = or_mask[0]
            or_mask_per_slot = or_mask[1]

            masks = self.__generate_mask(and_aggr_mask, or_aggr_mask)

            return masks, {'and': and_mask_per_slot, 'or': or_mask_per_slot}
        else:
            return self.__generate_mask(and_mask, or_mask)

    def generate_mask_batch(self, templates, **kwargs):
        and_mask = self.and_aggregator.generate_mask_batch(templates, **kwargs)
        or_mask = self.or_aggregator.generate_mask_batch(templates, **kwargs)

        return self.__generate_mask(and_mask, or_mask)

    def check(self, template, slot, target, **kwargs):
        return self.and_aggregator.check(template, slot, target, **kwargs)

    @property
    def and_maskers(self):
        return self.and_aggregator.maskers

    @property
    def or_maskers(self):
        return self.or_aggregator.maskers


def create_masker_params_from_jericho_env(info):
    masker_params = {
        'obs': info['state_description'],  # LMMasker
        'observed': info['observed_words'],  # ObservationMasker
        'observed_last': info['observed_words_last']  # LastObservationMasker
    }

    return masker_params


def create_masker_params_from_jericho_env_batch(infos, return_list=True):
    masker_params_list = [create_masker_params_from_jericho_env(i) for i in infos]
    keys = masker_params_list[0].keys()
    masker_params_dict = {k: [d[k] for d in masker_params_list] for k in keys}

    if return_list:
        return masker_params_dict, masker_params_list
    else:
        return masker_params_dict
