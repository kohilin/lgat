import os
import pickle
import logging

import numpy as np

from more_itertools import flatten
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

from jericho import FrotzEnv
from jericho.defines import ABBRV_DICT


logger = logging.getLogger(__name__)


INVALID_WORDS = ['script', 'recording', 'transcrip', 'reco']  # make Jericho game engine down


class JerichoEnv:
    def __init__(self, rom_file, seed=None, max_episode_steps=500, **kwargs):
        self.rom_file = rom_file
        self.env = FrotzEnv(self.rom_file, seed, **kwargs)
        self.max_episode_steps = max_episode_steps

        self.observed_words = set()
        self.observed_words_last = set()
        self.seen_state = set()

        self.last_action = None

        self.last = None
        self.step_count = 0

        self.history = []

        self.name = os.path.split(rom_file)[-1]

    def _create_info(self, info, action):
        info['world_changed'] = self.env._world_changed()

        state_hash = self.env.get_world_state_hash()

        info['is_unseen_state'] = int(state_hash not in self.seen_state)
        self.seen_state.add(state_hash)

        state = self.env.get_state()

        look, _, _, _ = self.env.step('look')
        self.env.set_state(state)

        try:
            ob = state[-1].decode('utf-8').strip()
        except UnicodeDecodeError as e:
            logger.info(f"Failed to decode: {state[-1]}")
            ob = str(state[-1])  # forcibly convert str
        inv = [o.name.lower() for o in self.env.get_inventory()]

        info["ob"] = ob.strip()
        info["look"] = look.strip()
        info["inv"] = inv
        info["game_file"] = self.rom_file

        inv_str = None
        if inv:
            if len(inv) == 1:
                inv_str = f' You have {inv[0]}.'
            else:
                inv_str = ' You have ' + ', '.join(inv[:-1]) + f' and {inv[-1]}.'
        else:
            inv_str = ' You have nothing.'

        info['state_description'] = info['look'] + ' ' + ob + ' ' + inv_str

        self._update_observed_words(info)

        info['observed_words'] = self.observed_words
        info['observed_words_last'] = self.observed_words_last

        return info

    def _update_observed_words(self, info):
        words = []

        for text in info['state_description'].split('|'):
            text = text.replace('\n', ' ')
            text = text.replace("\\n", ' ')
            text = text.replace("b'", '')
            text = text.lower()

            for w in word_tokenize(text):
                words.append(wn.morphy(w) or w)

        self.observed_words.update(words)
        self.observed_words_last = set(words)

    def reset(self):
        self.last = None
        self.step_count = 0

        self.observed_words = set()
        self.observed_words_last = set()
        self.seen_state = set()

        ob, info = self.env.reset()

        info = self._create_info(info, 'look')

        self.last_action = None

        self.step_count += 1
        self.history = []
        self.history.append((ob, None, self.score))

        return info['state_description'], info

    def step(self, action):
        for w in INVALID_WORDS:
            if w in action:
                action = action.replace(w, 'None')

        self.last_action = action

        state = self.env.get_state()

        if self.last is not None and self.last[2]:  # already finished
            ob, reward, done, info = self.last
        else:
            try:
                ob, reward, done, info = self.env.step(action)
            except RuntimeError as e:
                logger.error(f'RuntimeError: action={action} state={state}\n{e}')

        if self.step_count >= self.max_episode_steps:
            done = True

        info = self._create_info(info, action)

        self.last = ob, reward, done, info
        self.step_count += 1
        self.history.append((ob, action, self.score))

        return info['state_description'], reward, done, info

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def seed(self, seed=None):
        self.env.seed(seed)

    def is_change_world_state_hash(self, a):
        original_state = self.env.get_state()
        original_hash = self.env.get_world_state_hash()

        self.env.step(a)
        a_hash = self.env.get_world_state_hash()

        self.env.set_state(original_state)

        return a_hash != original_hash

    def be_same_transition(self, a, b):
        original_state = self.env.get_state()

        self.env.step(a)
        a_hash = self.env.get_world_state_hash()

        self.env.set_state(original_state)

        self.env.step(b)
        b_hash = self.env.get_world_state_hash()

        self.env.set_state(original_state)

        return a_hash == b_hash

    @property
    def score(self):
        return self.env.get_score()

    @property
    def max_word_length(self):
        return self.env.bindings['max_word_length']

    @property
    def vocab(self):
        return [ABBRV_DICT.get(w.word, w.word) for w in self.env.get_dictionary() if w.word != '']

    @property
    def walkthrough(self):
        return self.env.get_walkthrough()

    @property
    def world_state_hash(self):
        return self.env.get_world_state_hash()

    @property
    def dump(self):
        state = self.env.get_state()
        obs = self.env.frotz_lib.get_narrative_text().decode('utf-8')
        byte_state = pickle.dumps(state)
        return self.last_action, obs, byte_state


class JerichoBatchEnv:
    def __init__(self, rom_file, batch_size, **kwargs):
        self.envs = [JerichoEnv(rom_file, **kwargs) for i in range(batch_size)]
        self.batch_size = batch_size
        self.last = [None] * self.batch_size

    def seed(self, seed=None):
        rng = np.random.RandomState(seed)
        seeds = list(rng.randint(65635, size=self.batch_size))
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)
        return seeds

    def reset(self):
        self.last = [None] * self.batch_size
        obs, infos = [], []
        for env in self.envs:
            ob, info = env.reset()
            obs.append(ob)
            infos.append(info)

        return obs, infos

    def step(self, actions):
        assert len(actions) == len(self.envs)
        obs, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            ob, reward, done, info = env.step(action)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self.last[i] = env.last

        return obs, rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()

    def write_score_to_tensorboard(self, writer, e, suffix=''):
        scores = [l[3]['score'] for l in self.last]
        score_avg = np.mean(scores)
        score_min = np.min(scores).astype(np.int32)
        score_max = np.max(scores).astype(np.int32)

        writer.add_scalar(f'score/avg{suffix}', score_avg, e)
        writer.add_scalar(f'score/min{suffix}', score_min, e)
        writer.add_scalar(f'score/max{suffix}', score_max, e)
        return score_avg, score_min, score_max

    @staticmethod
    def from_log(rom_file, log_file):
        import pandas as pd
        df = pd.read_csv(log_file)
        env = JerichoBatchEnv(rom_file, len(df))
        for e, byte_state in zip(env.envs, df.byte_state):
            s = pickle.loads(eval(byte_state))
            e.env.set_state(s)
        return env

    @property
    def env(self):
        return self.envs[0]

    @property
    def scores(self):
        return [e.score for e in self.envs]

    @property
    def max_score_env(self):
        return max(self.envs, key=lambda x: x.score)

    @property
    def max_word_length(self):
        return self.env.max_word_length

    @property
    def observed_words(self):
        return [e.observed_words for e in self.envs]

    @property
    def observed_words_last(self):
        return [e.observed_words_last for e in self.envs]

    @property
    def vocab(self):
        return self.env.vocab

    @property
    def walkthrough(self):
        return self.env.walkthrough

    @property
    def world_state_hash(self):
        return [e.world_state_hash for e in self.envs]

    @property
    def dump(self):
        return [e.dump for e in self.envs]


class JerichoMultiGameBatchEnv(JerichoBatchEnv):
    def __init__(self, rom_files, **kwargs):
        self.envs = [JerichoEnv(rf, **kwargs) for rf in rom_files]
        self.batch_size = len(rom_files)
        self.last = [None] * self.batch_size

    def write_score_to_tensorboard(self, writer, e):

        for env, l in zip(self.envs, self.last):
            writer.add_scalar(f'score/{env.name}', l[3]['score'], e)

        scores = [l[3]['score'] for l in self.last]
        score_avg = np.mean(scores)
        score_min = np.min(scores).astype(np.int32)
        score_max = np.max(scores).astype(np.int32)
        return score_avg, score_min, score_max


    @property
    def vocab(self):
        return set(flatten([e.vocab for e in self.envs]))