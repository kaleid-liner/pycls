from .tf_builder import anystage, res_bottleneck_block
from .dynamic_net import DynamicAnyStage
from .dynamic_module import DynamicResBottleneckBlock
import keras.layers as layers
import keras
import tensorflow as tf
import numpy as np

import os
import re

from ppadb.client import Client as AdbClient


def get_representative_dataset(input_shape):
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, *input_shape)
            yield [data.astype(np.float32)]
    return representative_dataset


class LatencyPredictor:
    def __init__(self, output_dir, host="127.0.0.1", serial="", bin_path=""):
        self.err = 0.2
        self.output_dir = output_dir
        self.host = host
        self.remote_dir = "/data/local/tmp/elastic-search"
        self.serial = serial
        self.bin_path = bin_path
        self.dsp_bin_path = "/data/local/tmp/stretch-bin/benchmark_model_jianyu_only_dsp"

    def _gen_model(self, hw, w_in, w_out, stride, d):
        params = {"bot_mul": 0.25, "group_w": w_out // 4, "se_r": 0.25, "k": 3}

        model_name = "hw{}_win{}_wout{}_s{}_d{}.tflite".format(hw, w_in, w_out, stride, d)
        output_path = os.path.join(self.output_dir, model_name)

        if not os.path.exists(output_path):
            print("Generating {}".format(model_name))
            input_shape = (hw, hw, w_in)
            img_input = layers.Input(shape=input_shape)
            x = anystage(img_input, w_in, w_out, stride, d, res_bottleneck_block, "stage", params)

            model = keras.Model(img_input, x, name="test")

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = get_representative_dataset(input_shape)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            tflite_model = converter.convert()

            with open(output_path, "wb") as f:
                f.write(tflite_model)
        else:
            print("{} already generated".format(model_name))

        return model_name, output_path

    @staticmethod
    def _parse(res: str):
        pattern = "avg=(-?\d+(?:\.\d+)?)"
        for line in reversed(res.splitlines()):
            if line.startswith("Timings (microseconds):"):
                m = re.search(pattern, line)
                if m:
                    return float(m[1])

    def predict(self, hw, w_in, w_out, stride, d, device="cpu"):
        model_name, output_path = self._gen_model(hw, w_in, w_out, stride, d)

        use_gpu = "true" if device == "gpu" else "false"
        use_hexagon = "true" if device == "hexagon" else "false"

        client = AdbClient(self.host, port=5037)
        if self.serial:
            device = client.device(self.serial)
        else:
            device = client.devices()[0]

        remote_path = os.path.join(self.remote_dir, model_name)
        device.push(output_path, remote_path)

        bin_path = self.bin_path if device != "dsp" else self.dsp_bin_path 
        command = "taskset f0 {} --graph={} --use_gpu={} --use_hexagon={} --num_threads=4 --use_xnnpack=false --enable_op_profiling=true --max_delegated_partitions=100 --warmup_min_secs=0 --min_secs=0 --warmup_runs=5 --num_runs=50".format(bin_path, remote_path, use_gpu, use_hexagon)
        res = device.shell(command)

        return LatencyPredictor._parse(res)

    def get_flops(self, hw, w_in, w_out, stride, d):
        params = {"bot_mul": 0.25, "group_w": w_out // 4, "se_r": 0.25, "k": 3}
        size = hw
        cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
        cx = DynamicAnyStage.complexity(cx, w_in, w_out, stride, d, DynamicResBottleneckBlock, params)
        return cx["flops"]
