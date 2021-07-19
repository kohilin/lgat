import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent(nn.Module):
    def __init__(self, encoder, act_generator, explorer, memory,
                 optimizer, optimizer_params=None, exploration_bonus=False,
                 reward_clip=(-1, 1), template_masking=True):
        super().__init__()
        self.encoder = encoder
        self.act_generator = act_generator
        self.explorer = explorer
        self.memory = memory
        from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer
        self.is_priority = isinstance(self.memory, PrioritizedReplayBuffer)

        optimizer_params = optimizer_params or {}
        self.optimizer = optimizer(self.parameters(), **optimizer_params)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)

        self.exploration_bonus = exploration_bonus

        self.reward_clip = reward_clip

        self.__template_masking = template_masking
        self.__tmpl_mask_cache = {}

        self.replay_n = 0

        self.to(DEVICE)

    def act(self, obs):
        state = self.encoder(obs)
        return self.act_generator.generate(state)[0]

    def remember(self, **kwargs):
        self.memory.append(**kwargs)

    def forward(self, obs, **kwargs):
        state = self.encoder(obs)

        if self.__template_masking:
            tmpl_mask = []
            for o in obs:
                tmpl_mask.append(self.get_template_mask(o))
            kwargs['tmpl_mask'] = torch.stack(tmpl_mask).to(DEVICE)

        outputs = self.act_generator.forward(state, self.explorer, **kwargs)

        if self.__template_masking:
            self.__update_template_masking_cache(obs, outputs)

        return outputs, state

    def replay(self, batch_size):
        exps = self.memory.sample(batch_size)

        state = self.encoder([e[0]['state'] for e in exps]).to(DEVICE)
        next_state = self.encoder([e[0]['next_state'] for e in exps]).to(DEVICE)
        reward = torch.Tensor([e[0]['reward'] for e in exps]).to(DEVICE)
        action = torch.stack([e[0]['action'] for e in exps]).to(DEVICE)
        infos = [e[0]['info'] for e in exps]

        if self.exploration_bonus:
            explore_bonus = \
                torch.Tensor([i['is_unseen_state'] for i in infos]).to(DEVICE)
            reward = reward + explore_bonus

        if self.reward_clip:
            reward.clamp_(*self.reward_clip)

        td_loss = self.act_generator.compute_loss(action=action, state=state, next_state=next_state, reward=reward)

        if self.is_priority:
            self.memory.update_errors(reward.tolist())

        self.optimizer.zero_grad()
        td_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 40)  # 40 comes from tdqn
        self.optimizer.step()

        self.replay_n += 1

        return td_loss.detach().cpu().numpy()

    def save(self, path):
        params = {'parameters': self.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(open(path, 'rb'))

        self.load_state_dict(params['parameters'])
        self.optimizer.load_state_dict(params['optimizer'])

    def get_template_mask(self, o):
        if o not in self.__tmpl_mask_cache:
            self.__tmpl_mask_cache[o] = \
                torch.ones(self.act_generator.template_collection.template_num)
        return self.__tmpl_mask_cache[o]

    def __update_template_masking_cache(self, obs, outputs):
        assert len(obs) == outputs['action'].shape[0]

        slots = ['v', 'o1', 'p', 'o2']
        is_non_zero_mask = \
            [(outputs[s]['mask'] == 1).bool().any(dim=1) for s in slots]

        is_selectable_template = torch.stack(is_non_zero_mask).T.all(dim=1)
        t_acts = outputs['tmpl']['action']
        for o, is_ok, ta in zip(obs, is_selectable_template, t_acts):
            self.__tmpl_mask_cache[o][ta] = is_ok
