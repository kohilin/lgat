import copy
import logging


import torch
from torch.nn import Linear, LayerNorm, Module, ReLU, Sequential

from pfrl.utils.copy_param import synchronize_parameters
from pfrl.utils import evaluating

from .q_selector import MaxQ, RandomQ
from ..helper import DEVICE


def synchronize(model, tar_model):
    synchronize_parameters(src=model, dst=tar_model, method='hard')


class MLP(Module):
    def __init__(self, input_size, output_size, layer_num=2, hidden_size=200, activation=ReLU):
        super(MLP, self).__init__()
        layers = list()
        layers.append(Linear(input_size, hidden_size))
        for _ in range(layer_num - 1):
            layers.append(Linear(hidden_size, hidden_size))
            layers.append(LayerNorm(hidden_size))
            layers.append(activation())
        layers.append(Linear(hidden_size, output_size))

        self.__layers = Sequential(*layers)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.__layers(x)


class QLayerBase(Module):

    def __init__(self, state_size, action_size, sync_interval=1000, sync_method='soft'):
        super(QLayerBase, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.__target_q_layer = None
        self.__sync_interval = sync_interval
        self.__target_forward_count = 0
        self.__sync_method = sync_method

    def forward(self, state_repr):
        raise NotImplementedError

    def target_q(self, next_state_repr, reward, done, gamma):
        if self.__target_q_layer is None:
            self.__target_q_layer = [copy.deepcopy(self)]
            self.__target_q_layer[0].eval()

        if self.__sync_interval:
            target_q_layer = self.__target_q_layer[0]
        else:
            target_q_layer = self

        with evaluating(target_q_layer), torch.no_grad():
            _, next_q_values = target_q_layer.max_q(next_state_repr)

            if done is None:
                done = torch.zeros_like(reward, device=DEVICE)
            tar_q_values = reward + next_q_values * gamma * (1 - done)

            self.__target_forward_count += 1
            if self.__sync_interval and self.__target_forward_count % self.__sync_interval == 0:
                self.synchronize_with_target_q_layer()

        return tar_q_values

    def loss_q(self, action=None, state=None, next_state=None, reward=None,
               done=None, mask=None, loss_fnc=torch.nn.functional.smooth_l1_loss,
               gamma=0.95, *args, **kwargs):
        q = self.forward(state)
        cur_q_values = torch.index_select(q, 1, action).diagonal()
        tar_q_values = self.target_q(next_state, reward, done, gamma)
        if mask is None:
            mask = torch.ones_like(reward, device=DEVICE)
        loss = loss_fnc(mask * cur_q_values, mask * tar_q_values)
        return loss

    def max_q(self, state_repr):
        q = self.forward(state_repr)
        return MaxQ.select(q)

    def random_q(self, state_repr):
        q = self.forward(state_repr)
        return RandomQ.select(q)

    def synchronize_with_target_q_layer(self):
        logging.info('Parameter sync')
        synchronize_parameters(self, self.__target_q_layer[0], self.__sync_method, 1e-2)


class QLayer(QLayerBase):
    def __init__(self, state_size, action_size, layer_num=2, hidden_size=200, activation=ReLU, **kwargs):
        super(QLayer, self).__init__(state_size, action_size, **kwargs)
        self.mlp_q = MLP(state_size, action_size, layer_num, hidden_size, activation)

    def forward(self, x):
        return self.mlp_q(x)

    @classmethod
    def build(cls, name, state_size, action_size, **kwargs):
        layer_num = kwargs.pop('layer_num', 2)
        hidden_size = kwargs.pop('hidden_size', 200)
        activation = kwargs.pop('activation', ReLU)
        return QLayer(state_size, action_size,
                      layer_num, hidden_size, activation, **kwargs)
