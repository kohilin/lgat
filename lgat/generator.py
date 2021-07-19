from collections import OrderedDict

import torch
import numpy as np

from .q import QLayer, QAction, merge_q_layer_outputs
from .vocab import VocabDict


def build_q_layers_for_template_action(state_size, template_collection, **kwargs):
    q_layers = OrderedDict()
    q_layer_type = kwargs.pop('q_layer_type', 'dqn')
    q_layers['tmpl'] = QLayer.build(q_layer_type, state_size, template_collection.template_num, **kwargs)

    for slot_name in template_collection.slot_names:
        q_layers[slot_name] = QLayer.build(q_layer_type, state_size, template_collection.vocab_size, **kwargs)
    return q_layers


class StaticAction(QAction):
    def __init__(self, state_size, actions, **kwargs):
        vocab = VocabDict(actions, add_null=False)
        super(StaticAction, self).__init__(state_size=state_size, q_size=vocab.size, q_layer_num=1, vocab=vocab, **kwargs)


class SlotAction(QAction):
    def __init__(self, state_size, words, slot_num, **kwargs):
        vocab = VocabDict(words, add_null=False)
        super(SlotAction, self).__init__(state_size=state_size, q_size=vocab.size, q_layer_num=slot_num, vocab=vocab, **kwargs)


class VocabSlotAction(QAction):
    def __init__(self, state_size, words_list, **kwargs):
        if 'q_layers' not in kwargs:
            q_layers = OrderedDict()
            for idx, words in enumerate(words_list):
                q_layers[f's{idx}'] = QLayer.build('dqn', state_size=state_size, action_size=len(words))
            kwargs['q_layers'] = q_layers
        self.vocabs = [VocabDict(words, add_null=False) for words in words_list]
        super(VocabSlotAction, self).__init__(state_size=state_size, **kwargs)

    def generate(self, state_repr=None, explorer=None, action=None, **kwargs):
        if action is None:
            outputs = self.forward(state_repr, explorer)
            action = outputs['action']
        action_strs = []
        for idx, x in enumerate(action.transpose(0, 1)):
            action_strs.append(self.vocabs[idx].to_str_list(x, int_cast=True))
        action_strs = np.array(action_strs).T
        action_strs = [' '.join(a) for a in action_strs]
        return action_strs


class TemplateActionBase(QAction):
    def __init__(self, state_size, template_collection, **kwargs):
        self.template_collection = template_collection
        q_layers = build_q_layers_for_template_action(state_size, template_collection, **kwargs)

        super(TemplateActionBase, self).__init__(vocab=self.template_collection.vocab, q_layers=q_layers, **kwargs)

    def generate(self, state_repr=None, explorer=None, action=None, **kwargs):
        if action is None:
            outputs = self.forward(state_repr, explorer)
            action = outputs['action']
        return self.template_collection.generate(action)

    def _make_forward_outputs(self, outputs, return_type, merge, generate):
        if merge:
            outputs = merge_q_layer_outputs(outputs, return_type)
            if generate:
                outputs['generate'] = self.generate(action=outputs['action'])
            return outputs
        else:
            return outputs


class TemplateAction(TemplateActionBase):
    def __init__(self, state_size, template_collection, **kwargs):
        super(TemplateAction, self).__init__(state_size, template_collection, **kwargs)

    def forward(self, state_repr, explorer=None, return_type='pt',
                merge=True, generate=False, masks=None, targets=None,
                excludes=None, masker_params=None, **kwargs):
        outputs = OrderedDict()

        t_outs = super().forward(state_repr, explorer, targets=['tmpl'], masks=masks, merge=False)
        outputs.update(t_outs)

        template_indices = t_outs['tmpl']['action']
        masks = self.template_collection.generate_mask_batch(template_indices,masker_params)

        s_outs = super().forward(state_repr, explorer, excludes=['tmpl'],masks=masks, merge=False)
        outputs.update(s_outs)

        return self._make_forward_outputs(outputs, return_type, merge, generate)


class RecurrentTemplateAction(TemplateActionBase):
    def __init__(self, state_size, template_collection, **kwargs):
        super(RecurrentTemplateAction, self).__init__(state_size, template_collection, **kwargs)

    def forward(self, state_repr, explorer=None, return_type='pt',
                merge=True, generate=False, masks=None, targets=None,
                excludes=None, masker_params=None, tmpl_mask=None, **kwargs):
        outputs = OrderedDict()

        masks = {}
        if tmpl_mask is not None:
            masks['tmpl'] = tmpl_mask
        else:
            masks['tmpl'] = None

        t_outs = super().forward(state_repr, explorer, targets=['tmpl'],
                                 masks=masks, merge=False)
        outputs.update(t_outs)
        template_indices = t_outs['tmpl']['action']
        batch_size = state_repr.shape[0]
        if masker_params is None:
            masker_params = {'generated': [''] * batch_size}
        else:
            assert isinstance(masker_params, dict)
            masker_params['generated'] = [''] * batch_size

        for name in self.q_layer_names:
            if name == 'tmpl':
                continue

            masker_params['target_slot'] = [name]

            if tmpl_mask == 'auto':
                masks = {name: torch.stack([m[name][t_idx, :] for (t_idx, m) in zip(template_indices, all_masks)])}
            else:
                masks = \
                    self.template_collection.generate_mask_batch(template_indices,
                                                                 masker_params)

            s_outs = super().forward(state_repr, explorer, targets=[name], masks=masks, merge=False, generate=True)
            outputs.update(s_outs)

            for idx, a in enumerate(s_outs[name]['generate']):
                masker_params['generated'][idx] += a + ' '

        return self._make_forward_outputs(outputs, return_type, merge, generate)


class LGAT(RecurrentTemplateAction):
    def __init__(self, state_size,vocab, masker=None, **kwargs):
        from .action import GeneralTemplateCollection

        if isinstance(masker, list):
            masker = GeneralTemplateCollection.build_masker(vocab, masker)

        tmpl_collection = GeneralTemplateCollection(masker=masker, vocab=vocab)
        super(LGAT, self).__init__(state_size, tmpl_collection, **kwargs)


