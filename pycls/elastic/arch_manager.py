from .dynamic_net import DynamicAnyNet
from pycls.core.config import cfg
import random
import pycls.core.distributed as dist


class ArchManager:
    
    def __init__(self):
        p = DynamicAnyNet.get_params()
        self.widths = p["widths"]
        self.groups = p["groups"]
        self.additional_archs = cfg.TEST.ADDITIONAL_ARCHS

    def _sample(self, funcs, sync=True):
        while True:
            arch = {
                "widths": [funcs["w"](stage_candidates) for stage_candidates in self.widths],
                "groups": [funcs["g"](stage_candidates) for stage_candidates in self.groups],
            }
            if self.validate(arch):
                break

        if sync and cfg.NUM_GPUS > 1:
            arch = dist.broadcast_object(arch)
        
        return arch

    @staticmethod
    def validate(arch):
        for w, g in zip(arch["widths"], arch["groups"]):
            if (w % g != 0) or ((w // g) % 8 != 0):
                return False
        return True

    def sample_max(self):
        return self._sample({"w": max, "g": min})

    def sample_min(self):
        return self._sample({"w": min, "g": max})
        
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
    