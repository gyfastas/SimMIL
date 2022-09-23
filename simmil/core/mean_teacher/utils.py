# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Utility functions and classes"""

import sys
from PIL import ImageFilter
import random

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el)
                                                            for el in lst)

def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())
