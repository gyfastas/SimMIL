import argparse
import os
import torch
import torchvision.models as models
from .base import BaseParser

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class AssumptionParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("--assumption", type=str, default="std",
                                choices=["std", "count", "co"])
        self.parser.add_argument("--pos_targets", type=int, nargs="+",
                                default=8,
                                help="positive class, could be multiple (in co-occurance)")
        self.parser.add_argument("--pos_ratio", type=float, default=0.2,
                                help="pos ratio that at least a ratio of pos instances in a bag requried to form a positive bag")
        self.parser.add_argument("--max_pos_ratio", type=float, default=1.0,
                                help="maximum number of positive instances in a bag to form a positive bag.")