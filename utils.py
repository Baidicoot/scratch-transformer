import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def merge_from_dict(self, d):
        self.__dict__.update(d)