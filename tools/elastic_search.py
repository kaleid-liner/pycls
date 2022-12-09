import pycls.core.builders as builders
from pycls.elastic.arch_manager import MBBasedArchManger
from pycls.elastic.predictor import LatencyPredictor
import pycls.core.config as config
from pycls.core.config import cfg
from pycls.elastic.dynamic_net import DynamicAnyStage
from pycls.elastic.utils import make_divisible

import argparse
import sys

import pandas as pd


def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    parser.add_argument("--host", required=False, default="127.0.0.1", type=str)
    parser.add_argument("--serial", required=False, default="", type=str)
    parser.add_argument("--tmp_path", default="/home/eka/pycls/tflite_models", type=str)
    parser.add_argument("--output_csv", default="stage_candidates.csv", type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


env_args = parse_args()
config.load_cfg(env_args.cfg)
config.assert_cfg()
cfg.freeze()
arch_manager = MBBasedArchManger(False)
model = builders.build_model()

arch = arch_manager.random_sample()
model.set_active_subnet(**arch)

predictor = LatencyPredictor(env_args.tmp_path, env_args.host, env_args.serial)

strides = [1, 2, 2, 2]
hws = [56, 56, 28, 14]
w_ins = [64, 320, 608, 1216]

alpha = 0.4
beta = 0.8
gamma = 0.8
diff = 0.2

candidates = []
for stage in range(0, 4):
    w_in = w_ins[stage]
    s = strides[stage]
    hw = hws[stage]
    max_d, max_w = arch_manager.max_branch(stage)
    args = (hw, w_in, max_w, s, max_d)
    max_flops = predictor.get_flops(*args)
    max_cpu_lat = predictor.predict(*args, "cpu")
    max_gpu_lat = predictor.predict(*args, "gpu")
    if max_cpu_lat < max_gpu_lat:
        main_device = "cpu"
        sub_device = "gpu"
        max_lat = max_cpu_lat
    else:
        main_device = "gpu"
        sub_device = "cpu"
        max_lat = max_gpu_lat

    for d, w in arch_manager.iter_branch(stage):
        args = (hw, w_in, w, s, d)
        flops = predictor.get_flops(*args)
        if flops < alpha * max_flops:
            continue
        lat = predictor.predict(*args, main_device)
        if lat > gamma * max_lat:
            continue
        for sub_d, sub_w in arch_manager.iter_branch(stage):
            sub_args = (hw, w_in, sub_w, s, sub_d)
            sub_flops = predictor.get_flops(*sub_args)
            if sub_flops + flops < beta * max_flops:
                continue
            sub_lat = predictor.predict(*sub_args, sub_device)
            if sub_lat > gamma * max_lat:
                continue
            print("Stage candidate, ({}, {}) and ({}, {})", d, w, sub_d, sub_w)
            candidates.append({
                "s": stage,
                "d": d,
                "w": w,
                "sub_d": sub_d,
                "sub_w": sub_w,
                "flops": flops + sub_flops,
                "latency": max(lat, sub_lat),
                "main_flops": flops,
                "sub_flops": sub_flops,
                "lat": lat,
                "sub_lat": sub_lat,
                "max_flops": max_flops,
                "max_latency": max_lat,
                "main_device": main_device,
            })
            df = pd.DataFrame.from_dict(candidates)
            df.to_csv(env_args.output_csv, index=False)
