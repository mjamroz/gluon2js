from glob import glob
import numpy as np
from tensorflow.keras.models import load_model
from utils import imread, load_synset, check_error

CLASSES = load_synset()

def inference_keras(model_path, image_path="misc/image.jpg"):
    model = load_model(model_path)
    img = imread()
    if 'resnet' in model_path: # NOTICE THAT
        img = np.transpose(img, [0, 2, 3, 1]) # NHWC
    images = np.vstack([img])
    pred = model.predict(images, batch_size=1)
    idx = np.argmax(pred, axis=1)[0]
    check_error(pred[0], model_path)

    return CLASSES[idx]

if __name__ == '__main__':
    for m_path in glob("*.h5"):
        plant_id = int(inference_keras(m_path))
        print(f"{m_path}: {plant_id}")
        assert plant_id == 7022
