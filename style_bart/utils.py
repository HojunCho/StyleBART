import os
from os.path import join as pjoin
import random
from contextlib import contextmanager

import torch
import torch.nn as nn

import numpy as np

from omegaconf import OmegaConf
from hydra.utils import to_absolute_path, get_original_cwd

def to_absolute_path_recursive(config):
    if OmegaConf.is_list(config):
        for index, item in enumerate(config):
            if OmegaConf.is_config(item):
                to_absolute_path_recursive(item)
            if isinstance(item, str) and item.startswith('^/'):
                config[index] = to_absolute_path(item[2:])
    elif OmegaConf.is_dict(config):
        for key, value in config.items():
            if OmegaConf.is_config(value):
                to_absolute_path_recursive(value)
            if isinstance(value, str) and value.startswith('^/'):
                config[key] = to_absolute_path(value[2:])

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

@contextmanager
def chdir_root(*paths):
    original = os.getcwd()
    try:
        os.chdir(get_original_cwd())
        yield tuple(pjoin(original, path) for path in paths)
    finally:
        os.chdir(original)

@contextmanager
def release_memory(*modules):
    try: 
        for module in modules:
            module.cpu()
        torch.cuda.empty_cache()
        yield
    finally:
        for module in modules:
            module.cuda()

@contextmanager
def no_module_grad(module: nn.Module, exclude=(nn.Embedding,)):
    for m in module.modules():
        if isinstance(m, exclude):
            continue

        for parameters in m.parameters():
            parameters.requires_grad_(False)

    try:
        yield module
    finally:
        for m in module.modules():
            if isinstance(m, exclude):
                continue

            for parameters in m.parameters():
                parameters.requires_grad_(True)
