from pycls.CUDAKernelEnergyPredictor.predictor import ONNXConvKernelEnergyPredictor
from .utils import RawArch2IRConverter
import subprocess
import re
import torch


class EfficiencyPredictor:
    
    def __init__(self, converter: RawArch2IRConverter, policy="DEFAULT"):
        self.converter = converter
        self.predictor = ONNXConvKernelEnergyPredictor("V100")
        self.policy = policy

    def get_efficiency(self, sample):
        ir = self.converter.convert(sample)
        total_eng = 0
        total_lat = 0
        for kernel in ir:
            if kernel["type"].startswith("conv"):
                algos = self.predictor.predict(
                    image_size=kernel["hw"],
                    kernel_size=kernel["kernel_size"],
                    in_channels=kernel["in_channels"],
                    out_channels=kernel["out_channels"],
                    stride=kernel["stride"],
                    groups=kernel["groups"],
                )
                if self.policy == "DEFAULT":
                    efficiency = algos["CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"]
                elif self.policy == "ENERGY":
                    algo = min(algos.items(), key=lambda x: x[1]["energy"])
                    efficiency = algo[1]
            print(algo)
            total_eng += efficiency["energy"]
            total_lat += efficiency["latency"]
        return total_eng, total_lat


class MeasurementBasedPredictor:

    def __init__(self, bin_path: str, model_path: str, gpu_device: int = 0, warmup: int = 10, repeat: int = 1000):
        self.bin_path = bin_path
        self.model_path = model_path
        self.gpu_device = gpu_device
        self.warmup = warmup
        self.repeat = repeat
        self.dummy_input = torch.randn(1, 3, 224, 224)

    def get_efficiency(self, model):
        torch.onnx.export(model, self.dummy_input, self.model_path)
        cmd = f"CUDA_VISIBLE_DEVICES={self.gpu_device} {self.bin_path} -i {self.model_path} -w -x {self.warmup} -r {self.repeat} -d {self.gpu_device}"
        result = subprocess.check_output(cmd, shell=True)
        energy_str, latency_str = result.decode("utf-8").strip().splitlines()[-2:]
        energy_pattern = r"energies\(in vector\) : \(\d,([\d\.\d]+)\)"
        latency_pattern = r"gpu latency = ([\d\.\d]+) second"

        energy, latency = 0, 0
        m = re.match(energy_pattern, energy_str)
        if m:
            energy = float(m[1]) / self.repeat
        m = re.match(latency_pattern, latency_str)
        if m:
            latency = float(m[1]) / self.repeat

        return energy, latency
