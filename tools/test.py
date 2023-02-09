from nn_meter.ir_converter.onnx_converter import OnnxConverter
import onnx
import json


model = onnx.load("/data2/nvprof_onnx_models/regnetx-1_6gf.onnx")
converter = OnnxConverter(model)
ir = converter.convert()
json.dump(ir, open("regnetx-1_6gf.json", "w"))
