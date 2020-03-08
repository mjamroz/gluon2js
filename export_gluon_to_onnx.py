import numpy as np
import onnx as onnx_real
from mxnet import (image, cpu, init)
from mxnet.gluon import nn
from mxnet.contrib import onnx
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval
from utils import load_synset

SYNSET=load_synset()
CLASSES=len(SYNSET)
IMG = transform_eval(image.imread("misc/image.jpg"))
context = [cpu()]

def export_to_resnet101_v1(params_file, output="resnet101_v1"):
    net = get_model("ResNet101_v1", ctx=context, pretrained=True)
    net.cast(np.float16)
    with net.name_scope():
        net.output = nn.Dense(CLASSES)
    net.cast(np.float32)
    net.load_parameters(params_file, ctx=context, cast_dtype=True)
    net.hybridize(static_alloc=True, static_shape=True)
    net.collect_params().initialize()
    net(IMG)
    net.export(output)

def export_to_mobilenet10(params_file, output="mobilenet1.0"):
    net = get_model("mobilenet1.0", ctx=context, pretrained=True)
    net.cast(np.float16)
    with net.name_scope():
        net.output = nn.Dense(CLASSES)
    net.cast(np.float32)
    net.load_parameters(params_file, ctx=context, cast_dtype=True)
    net.hybridize(static_alloc=True, static_shape=True)
    net.collect_params().initialize()
    net(IMG)
    net.export(output)

def convert_to_onnx(prefix, cast=np.float32):
    sym = f"{prefix}-symbol.json"
    params = f"{prefix}-0000.params"
    path = onnx.export_model(sym, params, [(1,3,224,224)], cast, f"{prefix}.onnx")

    model = onnx_real.load_model(path)
    assert onnx_real.checker.check_graph(model.graph) is None

if __name__ == '__main__':
    export_to_resnet101_v1("models/resnet101_v1.params")
    export_to_mobilenet10("models/mobilenet1.0.params")
    convert_to_onnx("resnet101_v1")
    convert_to_onnx("mobilenet1.0")
