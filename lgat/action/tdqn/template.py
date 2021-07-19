import re

from collections import OrderedDict
from jericho import FrotzEnv
from jericho.defines import ABBRV_DICT

from ..masker import DummyMasker
from ..base import SlotBase, TemplateBase, TemplateCollectionBase
from ...vocab import VocabDict


class TDQNTemplate(TemplateBase):
    def __init__(self, template):
        slots = OrderedDict()

        self.num_of_obj = len(re.findall('OBJ', template))

        t_spl = template.split()
        obj_idxs = [i for (i, w) in enumerate(t_spl) if w == 'OBJ']
        if self.num_of_obj == 2:
            lm_suffix_o1 = ' '.join(t_spl[:obj_idxs[0]])
            lm_suffix_o2 = ' '.join(t_spl[:obj_idxs[1]]).replace('OBJ', 'it')
            slots['o1'] = SlotBase(name='o1', pos='n', lm_suffix=lm_suffix_o1, is_null=False, is_nullable=False)
            slots['o2'] = SlotBase(name='o2', pos='n', lm_suffix=lm_suffix_o2, is_null=False, is_nullable=False)
        elif self.num_of_obj == 1:
            lm_suffix_o1 = ' '.join(t_spl[:obj_idxs[0]])
            slots['o1'] = SlotBase(name='o1', pos='n', lm_suffix=lm_suffix_o1, is_null=False, is_nullable=False)
            slots['o2'] = SlotBase(name='o2', pos='n', lm_suffix='', is_null=True, is_nullable=True)
        else:
            slots['o1'] = SlotBase(name='o1', pos='n', lm_suffix='', is_null=True, is_nullable=True)
            slots['o2'] = SlotBase(name='o2', pos='n', lm_suffix='', is_null=True, is_nullable=True)

        super(TDQNTemplate, self).__init__(slots, name=template)

        self.original_template = template


class TDQNTemplateCollection(TemplateCollectionBase):
    def __init__(self, rom_file, use_maskers=False):
        slot_names = ['o1', 'o2']

        if isinstance(rom_file, list):
            templates, game_vocab, max_token_length = [], [], 0
            game_vocab = []
            for r in rom_file:
                env = FrotzEnv(r)
                templates += env.act_gen.templates
                game_vocab += [ABBRV_DICT.get(w.word, w.word) for w in env.get_dictionary() if w.word != '']
                if env.bindings['max_word_length'] > max_token_length:
                    max_token_length = env.bindings['max_word_length']
            templates = [TDQNTemplate(t) for t in set(templates)]
            game_vocab = set(game_vocab)
        else:
            env = FrotzEnv(rom_file)

            templates = [TDQNTemplate(t) for t in env.act_gen.templates]

            game_vocab = [ABBRV_DICT.get(w.word, w.word) for w in env.get_dictionary() if w.word != '']

            max_token_length=env.bindings['max_word_length']

        vocab = VocabDict(tokens=game_vocab, max_token_length=max_token_length)

        if use_maskers:
            from ..masker import MaskerBase, StopWordsMasker, PoSMasker, LMMasker, AndMaskAggregator
            MaskerBase.set_vocab(vocab)
            masker = AndMaskAggregator([StopWordsMasker(), PoSMasker(), LMMasker()])
        else:
            masker = DummyMasker()

        masker.set_vocab(vocab)

        super(TDQNTemplateCollection, self).__init__(templates, vocab, masker, slot_names)

    def _generate(self, t_idx, o1_idx, o2_idx):

        t = self.templates[t_idx]

        if t.num_of_obj == 2:
            o1_str = self.vocab.to_str(o1_idx)
            o2_str = self.vocab.to_str(o2_idx)
            return t.original_template.replace('OBJ', o1_str, 1).replace('OBJ', o2_str)
        elif t.num_of_obj == 1:
            o1_str = self.vocab.to_str(o1_idx)
            return t.original_template.replace('OBJ', o1_str)
        else:
            return t.original_template

    def generate(self, action):
        action_strs = []
        for a in action.cpu().detach().tolist():
            action_strs.append(self._generate(*a))

        return action_strs
