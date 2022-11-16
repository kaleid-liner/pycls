from .dynamic_net import DynamicAnyNet, DynamicRegNet, MBDynamicAnyNet
from pycls.core.config import cfg
import random
import numpy as np
import pycls.core.distributed as dist
import pycls.models.blocks as bk
from pycls.models.regnet import generate_regnet
from pycls.elastic.utils import make_divisible
import pycls.sweep.random as rand

import copy


class ArchManager:
    
    def __init__(self):
        p = DynamicAnyNet.get_params()
        self.widths = p["widths"]
        self.groups = p["groups"]
        self.additional_archs = cfg.TEST.ADDITIONAL_ARCHS

        max_arch = self.sample_max()
        self.group_widths = [w // g for w, g in zip(max_arch["widths"], max_arch["groups"])]

    def _sample(self, funcs, sync=True, validate=True):
        while True:
            arch = {
                "widths": [funcs["w"](stage_candidates) for stage_candidates in self.widths],
                "groups": [funcs["g"](stage_candidates) for stage_candidates in self.groups],
            }
            if not validate or self.validate(arch):
                break

        if sync and cfg.NUM_GPUS > 1:
            arch = dist.broadcast_object(arch)
        
        return arch

    def validate(self, arch, aligned=False):
        for w, g, gw in zip(arch["widths"], arch["groups"], self.group_widths):
            if (w % g != 0) or ((w // g) % 8 != 0):
                return False
            if aligned and (gw % (w // g) != 0):
                return False
        return True

    def sample_max(self):
        return self._sample({"w": max, "g": min}, validate=False)

    def sample_min(self):
        return self._sample({"w": min, "g": max}, validate=False)
        
    def random_sample(self):
        return self._sample({"w": random.choice, "g": random.choice})

    @property
    def len_archs(self):
        return len(self.additional_archs) + 2

    def iter_archs(self):
        yield self.sample_max()
        yield self.sample_min()
        for widths, groups in self.additional_archs:
            yield {"widths": widths, "groups": groups}


def generate_regnet_full(w_a, w_0, w_m, d, g):
    """Generates per stage ws, ds, gs, bs, and ss from RegNet cfg."""
    ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
    ss = [cfg.REGNET.STRIDE for _ in ws]
    bs = [cfg.REGNET.BOT_MUL for _ in ws]
    gs = [g for _ in ws]
    ws, bs, gs = bk.adjust_block_compatibility(ws, bs, gs)
    return ws, ds, ss, bs, gs

    
def validate_list_in_range(l, min_l, max_l):
    if not isinstance(min_l, list):
        for x in l:
            if x < min_l or x > max_l:
                return False
    else:
        for x, min_x, max_x in zip(l, min_l, max_l):
            if x < min_x or x > max_x:
                return False
    return True


def transpose_list(l):
    return [x[0] for x in l], [x[1] for x in l]


class RegNetBasedArchManager(ArchManager):
    def __init__(self, sync=True):
        p = DynamicRegNet.get_params()
        self.min_w_a, self.max_w_a = p["w_a_range"]
        self.min_w_0, self.max_w_0 = p["w_0_range"]
        self.min_w_m, self.max_w_m = p["w_m_range"]
        self.min_d, self.max_d = p["d_range"]
        self.min_ds, self.max_ds = transpose_list(p["ds_range"])
        self.min_g, self.max_g = p["g_range"]
        self.min_g_m, self.max_g_m = p["g_m_range"]
        self.min_ws, self.max_ws = transpose_list(p["ws_range"])
        
        self.sync = sync

    def _sample(self, arch):
        if self.sync and cfg.NUM_GPUS > 1:
            arch = dist.broadcast_object(arch)
        
        return arch

    def sample_max(self):
        return self._sample({
            "ws": self.max_ws,
            "ds": self.max_ds,
            "bs": [1.0 for _ in self.min_ws],
            "gs": [self.max_g for _ in self.min_ws],
        })

    def sample_min(self):
        return self._sample({
            "ws": self.min_ws,
            "ds": self.min_ds,
            "bs": [1.0 for _ in self.min_ws],
            "gs": [self.min_g for _ in self.min_ws],
        })

    def random_split_regnet_by_b(self):
        cfg.REGNET.BOT_MUL = 1.0
        arch_l = self.random_sample_regnet()
        cfg.REGNET.BOT_MUL = 0.5
        arch_r = self.random_sample_regnet()
        return arch_l, arch_r

    def random_split_regnet_by_wm(self):
        w_m_mid = np.exp(np.log(self.min_w_m + self.max_w_m) / 2)
        old_min_w_m = self.min_w_m
        old_max_w_m = self.max_w_m
        self.max_w_m = w_m_mid
        arch_l = self.random_sample_regnet()
        self.max_w_m = old_max_w_m
        self.min_w_m = w_m_mid
        arch_r = self.random_sample_regnet()
        self.min_w_m = old_min_w_m
        return arch_l, arch_r

    def random_split_regnet_by_wa(self):
        wa_mid = np.exp((np.log(self.min_w_a) + np.log(self.max_w_a) / 2))

        while True:
            w_a_l = rand.log_uniform(self.min_w_a, wa_mid, 0.1)
            w_a_r = rand.log_uniform(wa_mid, self.max_w_a, 0.1)
            w_0 = rand.log_uniform(self.min_w_0, self.max_w_0, 8)
            w_m = rand.log_uniform(self.min_w_m, self.max_w_m, 0.001)
            d = rand.uniform(self.min_d, self.max_d, 1)
            g = rand.log_uniform(self.min_g, self.max_g, 8)

            ws, ds, ss, bs, gs = generate_regnet_full(w_a_l, w_0, w_m, d, g)
            if len(ws) != 4:
                continue
            if not validate_list_in_range(ws, self.min_ws, self.max_ws):
                continue
            arch_l = self._sample({"ws": ws, "ds": ds, "bs": bs, "gs": gs})

            ws, ds, ss, bs, gs = generate_regnet_full(w_a_r, w_0, w_m, d, g)
            if len(ws) != 4:
                continue
            if not validate_list_in_range(ws, self.min_ws, self.max_ws):
                continue
            arch_r = self._sample({"ws": ws, "ds": ds, "bs": bs, "gs": gs})

            return arch_l, arch_r

    def random_sample_regnet(self):
        while True:
            w_a = rand.log_uniform(self.min_w_a, self.max_w_a, 0.1)
            w_0 = rand.log_uniform(self.min_w_0, self.max_w_0, 8)
            w_m = rand.log_uniform(self.min_w_m, self.max_w_m, 0.001)
            d = rand.uniform(self.min_d, self.max_d, 1)
            g = rand.log_uniform(self.min_g, self.max_g, 8)
            ws, ds, ss, bs, gs = generate_regnet_full(w_a, w_0, w_m, d, g)

            if len(ws) != 4:
                continue

            # validate d, w range
            if not validate_list_in_range(ds, self.min_ds, self.max_ds) or \
               not validate_list_in_range(ws, self.min_ws, self.max_ws):
                continue

            return self._sample({"ws": ws, "ds": ds, "bs": bs, "gs": gs})

    def random_sample(self, based_arch=None):
        while True:
            if based_arch is None:
                based_arch = self.random_sample_regnet()
            else:
                based_arch = copy.deepcopy(based_arch)
            ws, ds, bs, gs = based_arch["ws"], based_arch["ds"], based_arch["bs"], based_arch["gs"]
            g = gs[0]

            min_g = min(self.min_g_m * g, self.max_g)
            max_g = min(self.max_g_m * g, self.max_g)
            new_g = rand.uniform(min_g, max_g, 8)
            g_m = new_g / g

            gs[2] = gs[3] = new_g
            ws[2] /= g_m ** .25
            ws[1] *= g_m ** .25

            ws, bs, gs = bk.adjust_block_compatibility(ws, bs, gs)
            if not validate_list_in_range(ds, self.min_ds, self.max_ds) or \
               not validate_list_in_range(ws, self.min_ws, self.max_ws) or \
               not validate_list_in_range(gs, self.min_g, self.max_g):
                based_arch = None
                continue

            return self._sample({"ws": ws, "ds": ds, "bs": bs, "gs": gs})

    @property
    def len_archs(self):
        return 2

    def iter_archs(self):
        yield self.sample_max()
        yield self.sample_min()

    @staticmethod
    def arch_to_anynet_params(arch):
        return {
            "stem_type": cfg.REGNET.STEM_TYPE,
            "stem_w": cfg.REGNET.STEM_W,
            "block_type": cfg.REGNET.BLOCK_TYPE,
            "depths": arch["ds"],
            "widths": arch["ws"],
            "strides": [cfg.REGNET.STRIDE] * 4,
            "bot_muls": arch["bs"],
            "group_ws": arch["gs"],
            "head_w": cfg.REGNET.HEAD_W,
            "se_r": cfg.REGNET.SE_R if cfg.REGNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }


class MBBasedArchManger(RegNetBasedArchManager):
    def __init__(self, sync=True):
        p = MBDynamicAnyNet.get_params()
        self.min_dm, self.max_dm = p["dm_range"]
        self.min_wm, self.max_wm = p["wm_range"]
        self.max_ds = p["depths"]
        self.max_ws = p["widths"]
        self.bot_muls = p["bot_muls"]
        self.max_gs = p["group_ws"]
        self.n_branch = p["n_branch"]

        self.min_ds = [make_divisible(d * self.min_dm, 1) for d in self.max_ds]
        self.min_gs = [make_divisible(g * self.min_wm, 8) for g in self.max_gs]
        self.min_ws = self._gs_to_ws(self.min_gs)
        
        self.sync = sync

    def _list_to_mblist(self, l):
        return list(zip(*(l for _ in range(self.n_branch))))
    
    def _gs_to_ws(self, gs):
        return [make_divisible(g / b, 8) for g, b in zip(gs, self.bot_muls)]

    def sample_max(self):
        return self._sample({
            "wss": self._list_to_mblist(self.max_ws),
            "dss": self._list_to_mblist(self.max_ds),
            "bss": self._list_to_mblist(self.bot_muls),
            "gss": self._list_to_mblist(self.max_gs),
        })

    def sample_min(self):
        return self._sample({
            "wss": self._list_to_mblist(self.min_ws),
            "dss": self._list_to_mblist(self.min_ds),
            "bss": self._list_to_mblist(self.bot_muls),
            "gss": self._list_to_mblist(self.min_gs),
        })

    def random_sample(self):
        dss = []
        bss = []
        gss = []
        wss = []
        for _ in range(self.n_branch):
            dss.append([rand.uniform(min_d, max_d, 1) for min_d, max_d in zip(self.min_ds, self.max_ds)])
            bss.append(self.bot_muls)
            gs = [rand.log_uniform(min_g, max_g, 8) for min_g, max_g in zip(self.min_gs, self.max_gs)]
            gss.append(gs)
            wss.append(self._gs_to_ws(gs))
        return self._sample({
            "wss": list(zip(*wss)),
            "dss": list(zip(*dss)),
            "bss": list(zip(*bss)),
            "gss": list(zip(*gss)),
        })
