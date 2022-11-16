#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

import itertools

from pycls.core.config import cfg
from pycls.models.blocks import init_weights
from torch.nn import Module
import torch
from .dynamic_module import DynamicResBottleneckBlock, DynamicSimpleStem, DynamicAnyHead, DynamicModule, DynamicResStem
from .utils import make_divisible


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "simple_stem_in": DynamicSimpleStem,
        "res_stem_in": DynamicResStem,
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
        self.active_depth = d
        for i in range(d):
            block = block_fun(w_in_list, w_out_list, stride, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in_list = 1, w_out_list

    def forward(self, x):
        for block in itertools.islice(self.children(), self.active_depth):
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

    def set_active_subnet(self, w, **kwargs):
        super().set_active_subnet(w, **kwargs)
        self.active_depth = kwargs["d"]

        
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


class DynamicRegNet(DynamicModule):
    
    @staticmethod
    def get_params():
        return {
            "w_a_range": cfg.DYNAMICREGNET.WA_RANGE,
            "w_0_range": cfg.DYNAMICREGNET.W0_RANGE,
            "w_m_range": cfg.DYNAMICREGNET.WM_RANGE,
            "d_range": cfg.DYNAMICREGNET.D_RANGE,
            "g_range": cfg.DYNAMICREGNET.G_RANGE,
            "g_m_range": cfg.DYNAMICREGNET.GM_RANGE,
            "ws_range": cfg.DYNAMICREGNET.WS_RANGE,
            "ds_range": cfg.DYNAMICREGNET.DS_RANGE,
            "stem_type": cfg.REGNET.STEM_TYPE,
            "stem_w": cfg.DYNAMICREGNET.STEM_W,
            "block_type": cfg.REGNET.BLOCK_TYPE,
            "strides": [cfg.REGNET.STRIDE] * 4,
            "bot_muls": [cfg.REGNET.BOT_MUL] * 4,
            "head_w": cfg.REGNET.HEAD_W,
            "se_r": cfg.REGNET.SE_R if cfg.REGNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, params=None):
        super().__init__()
        p = DynamicRegNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["ds_range", "ws_range", "strides", "bot_muls"]
        g = p["g_range"][1]
        for i, (d, w, s, b) in enumerate(zip(*[p[k] for k in keys])):
            d = d[1]
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            stage = DynamicAnyStage(prev_w, w, s, d, block_fun, params)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        self.head = DynamicAnyHead(prev_w, p["head_w"], p["num_classes"])
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def set_active_subnet(self, ws, ds, bs, gs):
        for i, (w, d, b, g) in enumerate(zip(ws, ds, bs, gs)):
            getattr(self, "s{}".format(i + 1)).set_active_subnet(w=w, d=d, b=b, g=g)

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
            params = {"bot_mul": b, "min_groups": g, "se_r": p["se_r"]}
            ir, hw = DynamicAnyStage.gen_ir(ir, hw, prev_w, w, s, d, block_fun, params)
            prev_w = w
        ir, hw = DynamicAnyHead.gen_ir(ir, hw, prev_w, p["head_w"], [["num_classes"]])
        return ir


class MBDynamicAnyNet(DynamicModule):
    
    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {
            "dm_range": cfg.MBDYNAMICANYNET.DM_RANGE,
            "wm_range": cfg.MBDYNAMICANYNET.WM_RANGE,
            "n_branch": cfg.MBDYNAMICANYNET.N_BRANCH,
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.DYNAMICANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "head_w": cfg.ANYNET.HEAD_W,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
            "n_branch": cfg.MBDYNAMICANYNET.N_BRANCH,
        }

    def __init__(self, params=None):
        super().__init__()
        p = MBDynamicAnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        
        self.n_stage = len(cfg.ANYNET.DEPTHS)
        self.n_branch = p["n_branch"]
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            w = [make_divisible(w * wm, 8) for wm in p["wm_range"]]
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"], "vanilla_conv": True}
            for j in range(self.n_branch):
                stage = DynamicAnyStage(prev_w, w, s, d, block_fun, params)
                self.add_module("s{}_b{}".format(i + 1, j), stage)
            prev_w = [2 * _w for _w in w]
        self.head = DynamicAnyHead(prev_w, p["head_w"], p["num_classes"])
        self.apply(init_weights)

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.n_stage):
            x = torch.cat([
                getattr(self, "s{}_b{}".format(i + 1, j))(x)
                for j in range(self.n_branch)
            ], 1)
        x = self.head(x)
        return x

    def set_active_subnet(self, wss, dss, bss, gss):
        for i, (ws, ds, bs, gs) in enumerate(zip(wss, dss, bss, gss)):
            for j, (w, d, b, g) in enumerate(zip(ws, ds, bs, gs)):
                getattr(self, "s{}_b{}".format(i + 1, j)).set_active_subnet(w=w, d=d, b=b, g=g)
