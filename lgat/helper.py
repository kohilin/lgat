import os
import yaml

from collections import OrderedDict

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(ROOT, 'resources')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DIRECTION_WORDS = ["up", "down", "north", "south", "east", "west", "northwest", "northeast", "southwest", "southeast"]


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_representer(OrderedDict, represent_odict)
yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)


def dump_yaml(d, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(d, f)


def load_yaml(filepath):
    with open(filepath) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
