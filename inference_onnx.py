from glob import glob
import numpy
import onnxruntime

from utils import load_synset, imread, postprocess, check_error

CLASSES = load_synset()

def inference_onnx(model_path, image_path="misc/image.jpg"):
    input_data = imread(image_path)
    session = onnxruntime.InferenceSession(model_path, None)
    raw_result = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: input_data}
    )
    res = postprocess(raw_result)
    idx = numpy.argmax(res)
    check_error(raw_result[0][0], model_path)
    return CLASSES[idx]

if __name__ == '__main__':
    for model in glob("*.onnx"):
        plant_id = int(inference_onnx(model))
        print(f"{model}: {plant_id}")
        assert plant_id == 7022
