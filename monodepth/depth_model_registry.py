#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from .depth_model import DepthModel
from .midas_v2_model import MidasV2Model

from typing import List


class GTOptDepthModel:
    learning_rate: float = 0

    def eval(self):
        pass

    def parameters(self):
        return []

    def forward(self, x):
        return x * 0


def get_depth_model_list() -> List[str]:
    return ["midas2", "gtopt"]


def get_depth_model(type: str) -> DepthModel:
    if type == "midas2":
        return MidasV2Model
    elif type == "gtopt":
        return GTOptDepthModel
    else:
        raise ValueError(f"Unsupported model type '{type}'.")


def create_depth_model(type: str) -> DepthModel:
    model_class = get_depth_model(type)
    return model_class()
