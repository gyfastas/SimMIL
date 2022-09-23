import argparse
import os
import torch
import torchvision.models as models
from .assumption import AssumptionParser

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class InsMILParser(AssumptionParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("--pos_class", type=int, default=8,
                                help="Logit dimension that is used to convert multi-classifcation into binary")
        self.parser.add_argument("--ins_aggregation", type=str, default="prob_max",
                                choices=["prob_max", "prob_mean", "major_vote"])

    def parse_args(self, args=None):
        return self.parser.parse_args() if args is None else self.parser.parse_args(args)