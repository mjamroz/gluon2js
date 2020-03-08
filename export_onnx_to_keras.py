from glob import glob
from onnx2keras import onnx_to_keras
import onnx
import onnxruntime
import tensorflowjs as tfjs
import numpy
from inference_keras import inference_keras

def onnxtokeras(model_path="resnet101_v1.onnx"):
    keras_model = onnx_to_keras(
        onnx.load(model_path),
        [onnxruntime.InferenceSession(model_path, None).get_inputs()[0].name],
        change_ordering='resnet' in model_path, # NOTICE THAT
        name_policy='short',
        verbose=False
    )
    keras_model.save(model_path.replace(".onnx", ".h5"))
    return keras_model

def onnxtotfjs(model_path):
    k_model = onnxtokeras(model_path)
    tfjs.converters.save_keras_model(k_model, model_path.replace(".onnx", "_tfjs"))

if __name__ == '__main__':
    for path in glob("*.onnx"):
        onnxtotfjs(path)
        plant_id = int(inference_keras(path.replace(".onnx", ".h5")))
        print(f"Keras {path}: {plant_id}")
        assert plant_id == 7022
