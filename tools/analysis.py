import pycls.core.builders as builders
from pycls.elastic.arch_manager import RegNetBasedArchManager
from pycls.predictor import MeasurementBasedPredictor
from pycls.core.config import cfg
from pycls.sweep.random import quantize
import pycls.core.config as config
from pycls.models.anynet import AnyNet
from pycls.core import net

import argparse
import sys

import pandas as pd
import numpy as np


def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    parser.add_argument("--dimension", type=str, default="g_m")
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--bin_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    config.load_cfg(args.cfg)
    config.assert_cfg()
    cfg.freeze()

    arch_manager = RegNetBasedArchManager(False)
    predictor = MeasurementBasedPredictor(args.bin_path, args.model_path, args.gpu_id)

    df_dict = []
    output = args.output or "{}.csv".format(args.dimension)

    if args.dimension == "g_m":
        dim_range = np.arange(1.0, 2.1, 0.1)
    elif args.dimension == "w_a":
        dim_range = np.arange(arch_manager.min_w_a, arch_manager.max_w_a + 0.1, 2.0)
    elif args.dimension == "w_m":
        dim_range = np.arange(arch_manager.min_w_m, arch_manager.max_w_m + 0.1, 0.05)

    for _ in range(args.n_sample):
        raw_arch = {}
        arch_manager.random_sample_regnet(raw_arch=raw_arch, validate_range=False)
        for dim in dim_range:
            if args.dimension == "g_m":
                g2 = quantize(dim * raw_arch["g"], 8)
                raw_arch["g2"] = g2
            elif args.dimension == "w_a":
                raw_arch["w_a"] = quantize(dim, 0.1)
            elif args.dimension == "w_m":
                raw_arch["w_m"] = quantize(dim, 0.001)
            arch = arch_manager.random_sample(based_raw_arch=raw_arch, retry=False, validate_range=False)
            if arch is None:
                continue
            params = arch_manager.arch_to_anynet_params(arch)
            model = AnyNet(params)
            print("Model for arch {} generated. Running...".format(arch))

            size = cfg.TRAIN.IM_SIZE
            cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
            cx = model.complexity(cx, params)
            energy, latency = predictor.get_efficiency(model)

            df_dict.append({
                args.dimension: dim,
                "energy": energy,
                "latency": latency,
                **raw_arch,
                **cx,
                **arch,
            })
            df = pd.DataFrame.from_dict(df_dict)
            df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
