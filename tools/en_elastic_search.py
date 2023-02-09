import pycls.core.builders as builders
from pycls.elastic.arch_manager import RegNetBasedArchManager
from pycls.predictor import FLOPsBasedAccuracyPredictor, EfficiencyPredictor
from pycls.predictor.utils import RawArch2IRConverter
from pycls.core.config import cfg
import pycls.core.config as config
from pycls.elastic.search import EvolutionFinder

import argparse
import sys

import pandas as pd


def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


env_args = parse_args()
config.load_cfg(env_args.cfg)
config.assert_cfg()
cfg.freeze()
arch_manager = RegNetBasedArchManager(False)

converter = RawArch2IRConverter(arch_manager)
efficiency_predictor = EfficiencyPredictor(converter, policy="ENERGY")
accuracy_predictor = FLOPsBasedAccuracyPredictor(converter)

finder = EvolutionFinder(efficiency_predictor, accuracy_predictor, arch_manager)
finder.run_evolution_search(0, verbose=True)
