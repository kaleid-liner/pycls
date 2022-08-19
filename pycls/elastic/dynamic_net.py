#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

from pycls.core.config import cfg
from pycls.models.blocks import init_weights
from torch.nn import Module
from .dynamic_module import DynamicResBottleneckBlock, DynamicSimpleStem, DynamicAnyHead, DynamicModule


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "simple_stem_in": DynamicSimpleStem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "res_bottleneck_block": DynamicResBottleneckBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class DynamicAnyStage(DynamicModule):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in_list, w_out_list, stride, d, block_fun, params):
        super().__init__()
        for i in range(d):
            block = block_fun(w_in_list, w_out_list, stride, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in_list = 1, w_out_list

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x
    
    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, block_fun, params):
        for _ in range(d):
            cx = block_fun.complexity(cx, w_in, w_out, stride, params)
            stride, w_in = 1, w_out
        return cx

    @staticmethod
    def gen_ir(ir, hw, w_in, w_out, stride, d, block_fun, params):
        for _ in range(d):
            ir, hw = block_fun.gen_ir(ir, hw, w_in, w_out, stride, params)
            stride, w_in = 1, w_out
        return ir, hw

        
class DynamicAnyNet(DynamicModule):
    
    @staticmethod
    def get_params():
        nones = [None for _ in cfg.DYNAMICANYNET.DEPTHS]
        return {
            "stem_type": cfg.DYNAMICANYNET.STEM_TYPE,
            "stem_w": cfg.DYNAMICANYNET.STEM_W,
            "block_type": cfg.DYNAMICANYNET.BLOCK_TYPE,
            "depths": cfg.DYNAMICANYNET.DEPTHS,
            "widths": cfg.DYNAMICANYNET.WIDTHS,
            "strides": cfg.DYNAMICANYNET.STRIDES,
            "bot_muls": cfg.DYNAMICANYNET.BOT_MULS if cfg.DYNAMICANYNET.BOT_MULS else nones,
            "groups": cfg.DYNAMICANYNET.GROUPS if cfg.DYNAMICANYNET.GROUPS else nones,
            "head_w": cfg.DYNAMICANYNET.HEAD_W,
            "se_r": cfg.DYNAMICANYNET.SE_R if cfg.DYNAMICANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super().__init__()
        p = DynamicAnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "groups"]
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {"bot_mul": b, "min_groups": min(g), "se_r": p["se_r"]}
            stage = DynamicAnyStage(prev_w, w, s, d, block_fun, params)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = DynamicAnyHead(prev_w, p["head_w"], p["num_classes"])
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def set_active_subnet(self, widths=None, groups=None):
        for i, (w, g) in enumerate(zip(widths, groups)):
            getattr(self, "s{}".format(i + 1)).set_active_subnet(w=w, g=g)

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = DynamicAnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        cx = stem_fun.complexity(cx, 3, max(p["stem_w"]))
        prev_w = max(p["stem_w"])
        keys = ["depths", "widths", "strides", "bot_muls", "groups"]
        for d, w, s, b, g in zip(*[p[k] for k in keys]):
            params = {"bot_mul": b, "min_groups": min(g), "se_r": p["se_r"]}
            w = max(w)
            cx = DynamicAnyStage.complexity(cx, prev_w, w, s, d, block_fun, params)
            prev_w = w
        cx = DynamicAnyHead.complexity(cx, prev_w, p["head_w"], p["num_classes"])
        return cx

    @staticmethod
    def gen_ir(ir=None, hw=224, widths=None, groups=None, params=None):
        if ir is None:
            ir = []
        p = DynamicAnyNet.get_params() if not params else params
        p["widths"] = widths or [max(w) for w in p["widths"]]
        p["groups"] = groups or [min(g) for g in p["groups"]]
        p["stem_w"] = max(p["stem_w"])
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        ir, hw = stem_fun.gen_ir(ir, hw, 3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "groups"]
        for d, w, s, b, g in zip(*[p[k] for k in keys]):
            params = {"bot_mul": b, "groups": g, "se_r": p["se_r"]}
            ir, hw = DynamicAnyStage.gen_ir(ir, hw, prev_w, w, s, d, block_fun, params)
            prev_w = w
        ir, hw = DynamicAnyHead.gen_ir(ir, hw, prev_w, p["head_w"], [["num_classes"]])
        return ir
