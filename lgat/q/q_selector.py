import torch


def sanity_check_mask(mask):
    if not all(torch.sum(mask, dim=1)):
        raise ValueError(f"A zero mask (i.e., all elements are zero) was found. At least one element should be 1.")


def apply_mask(q, mask):
    q_ = q.clone()
    q_[mask == 0] = -1e10  # assign very small values to masked indices
    return q_


class Selector(object):
    @classmethod
    def select(cls, *args, **kwargs):
        raise NotImplementedError


class MaxQ(Selector):
    @classmethod
    def select(cls, q, mask=None):
        if mask is not None:
            q = apply_mask(q, mask)

        q_idxs = torch.argmax(q, dim=1)
        q_values = torch.index_select(q, 1, q_idxs).diagonal()
        return q_idxs, q_values


class RandomQ(Selector):
    @classmethod
    def select(cls, q, mask=None):
        if mask is not None:
            q_ = mask.float()
            zero_mask_rows = (q_ == 0).all(dim=1)
            q_[zero_mask_rows] = \
                torch.ones(q_.shape[1], device=q_.device)

        else:
            q_ = torch.ones_like(q)

        q_idxs = q_.multinomial(1).view(-1)

        q_values = torch.index_select(q, 1, q_idxs).diagonal()
        return q_idxs, q_values


class ProbQ(Selector):
    @classmethod
    def select(cls, q, mask=None):
        if mask is not None:
            q = apply_mask(q, mask)

        softmax_q = torch.softmax(q, dim=1)
        q_idxs = softmax_q.multinomial(1).view(-1)
        q_values = torch.index_select(q, 1, q_idxs).diagonal()
        return q_idxs, q_values

