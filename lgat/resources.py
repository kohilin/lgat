import os
import pickle

from more_itertools import flatten
from nltk.corpus import wordnet as wn


ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(ROOT, 'resources')


def load_line_by_line_file(filepath):
    with open(filepath) as f:
        return [l.strip() for l in f]


def load_prepositions():
    filepath = os.path.join(RESOURCE_PATH, 'prepositions.txt')
    return set(load_line_by_line_file(filepath))


def load_wordnet_word_list(v_or_n, refresh=False):
    dump_path = os.path.join(RESOURCE_PATH, f'wordnet-{v_or_n}.dump')
    if os.path.exists(dump_path) and not refresh:
        with open(dump_path, 'rb') as f:
            return pickle.load(f)

    synsets = wn.all_synsets(v_or_n)
    lemmas = flatten([[l.name() for l in s.lemmas()] for s in synsets])
    lemmas = set([l.replace('_', '') for l in lemmas])

    with open(dump_path, 'wb') as f:
        pickle.dump(lemmas, f)

    return lemmas


def load_stopwords():
    return load_line_by_line_file(os.path.join(RESOURCE_PATH, 'stopwords.txt'))
