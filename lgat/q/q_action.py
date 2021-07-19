from collections import OrderedDict

import torch
from torch import nn

from .q_net import QLayer
from .q_selector import MaxQ
from ..vocab import VocabDict


def convert_tensors(return_type, *tensors):
    if return_type == 'pt':
        return tensors
    elif return_type == 'np':
        return [t.detach().cpu().numpy() for t in tensors]
    elif return_type == 'list':
        return [t.detach().cpu().tolist() for t in tensors]


def merge_q_layer_outputs(dic, return_type):
    action, q_values, masks, qs = [], [], [], []
    for name, d in dic.items():
        action.append(d['action'].view(-1, 1))
        q_values.append(d['q_values'].view(-1, 1))
        qs.append(d['q'])
        masks.append(d['mask'])

        if return_type != 'pt':
            dic[name]['action'], dic[name]['q_values'], dic[name]['q'] = convert_tensors(return_type, action[-1], q_values[-1], qs[-1])

    action = torch.cat(action, dim=1)
    q_values = torch.cat(q_values, dim=1)

    action, q_values = convert_tensors(return_type, action, q_values)
    qs = convert_tensors(return_type, *qs)

    dic['action'] = action
    dic['q_values'] = q_values
    dic['q'] = qs
    dic['masks'] = masks

    return dic


class QAction(nn.Module):
    def __init__(self, state_size=None, q_size=None, q_layer_num=None, q_layer_names=None,
                 q_layer_type='dqn', q_layers=None, vocab=None, **kwargs):
        super(QAction, self).__init__()
        if q_layers is None:
            if q_layer_names is None:
                q_layer_names = [f'q{i}' for i in range(q_layer_num)]

            q_layers = OrderedDict()
            for name in q_layer_names:
                q_layers[name] = QLayer.build(q_layer_type, state_size, q_size, **kwargs)

        self.q_layers = nn.ModuleDict(OrderedDict(q_layers))

        if vocab is not None:
            if isinstance(vocab, list):
                vocab = VocabDict(vocab, add_null=False)

        self.vocab = vocab

    def forward(self, state_repr, explorer=None, return_type='pt',
                merge=True, generate=False, masks=None, targets=None,
                excludes=None, **kwargs):
        rtn_dict = OrderedDict()

        rt = 'pt' if merge else return_type

        for name, q_layer in self.q_layers.items():
            if targets and name not in targets:
                continue

            if excludes and name in excludes:
                continue

            q = q_layer(state_repr)

            mask = masks and masks.get(name, None)

            action_and_q_values = \
                explorer.select(q, mask) if explorer else MaxQ.select(q, mask)

            action_and_q_values = convert_tensors(rt, *action_and_q_values)
            q = convert_tensors(rt, q)[0]

            rtn_dict[name] = {'action': action_and_q_values[0], 'q_values': action_and_q_values[1], 'q': q, 'mask': mask}

            if generate and not merge:
                rtn_dict[name]['generate'] = self.generate(action=action_and_q_values[0].view(-1, 1))

        if merge:
            rtn_dict = merge_q_layer_outputs(rtn_dict, return_type)

            if generate:
                rtn_dict['generate'] = self.generate(action=rtn_dict['action'])

        return rtn_dict

    def generate(self, state_repr=None, explorer=None, action=None, **kwargs):
        if action is None:
            outputs = self.forward(state_repr, explorer)
            action = outputs['action']

        action_strs = self.vocab.to_str_batch(action, int_cast=True)

        action_strs = [' '.join(a) for a in action_strs]

        return action_strs

    def compute_loss(self, action, state, next_state, reward, gamma=0.95, loss_fnc=torch.nn.functional.smooth_l1_loss, **kwargs):
        losses = list()
        action_t = action.T
        for action, q_layer in zip(action_t, self.q_layers.values()):
            loss = q_layer.loss_q(action, state, next_state, reward, gamma=gamma, loss_fnc=loss_fnc, **kwargs)
            losses.append(loss)

        loss = torch.stack(losses).mean()

        return loss

    @property
    def q_layer_num(self):
        return len(self.q_layers)

    @property
    def q_layer_names(self):
        return list(self.q_layers.keys())
