import numpy as np

import torch

from .q_selector import Selector, MaxQ, RandomQ, ProbQ
from ..helper import DEVICE


class Explorer(Selector):
    def __init__(self, auto_decay=False, auto_decay_interval=10):
        self.__auto_decay=auto_decay
        self.__auto_decay_interval = auto_decay_interval
        self.__n = 0

    def increment(self):
        self.__n += 1
        if self.__auto_decay and self.__n % self.__auto_decay_interval == 0:
            self.update()

    def select(self, q):
        raise NotImplementedError

    def update(self):
        pass

    @classmethod
    def build(cls, params):
        method = params.pop('method', 'egreedy')
        if method == 'egreedy':
            return EGreedy(params.pop('eps', 1), params.pop('eps_min', 0.2), params.pop('eps_decay', 0.95))
        elif method == 'boltzmann':
            return Boltzmann()
        else:
            assert False


class EGreedy(Explorer):
    def __init__(self, eps=1.0, eps_min=0.2, eps_decay=0.995, **kwargs):
        super(EGreedy, self).__init__(**kwargs)
        self.__eps = eps
        self.__eps_min = eps_min
        self.__eps_decay = eps_decay

    def select(self, q, mask=None):
        max_q_indices, max_q_values = MaxQ.select(q, mask)
        random_q_indices, random_q_values = RandomQ.select(q, mask)

        batch_size = q.shape[0]

        rand_num = torch.rand(batch_size, device=DEVICE)

        lt_eps = (rand_num < self.__eps).long()
        gt_eps = (1.0 - lt_eps).long()

        q_idxs = gt_eps * max_q_indices + lt_eps * random_q_indices
        selected_q_values = gt_eps * max_q_values + lt_eps * random_q_values

        self.increment()

        return q_idxs, selected_q_values

    def update(self):
        if self.__eps > self.__eps_min:
            self.__eps *= self.__eps_decay

            if self.__eps < self.__eps_min:
                self.__eps = self.__eps_min

    @property
    def eps(self):
        return self.__eps

    @property
    def eps_min(self):
        return self.__eps_min

    @property
    def eps_decay(self):
        return self.__eps_decay

