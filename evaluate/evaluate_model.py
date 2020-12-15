from tensorflow.keras.models import load_model
import numpy as np
from model.base_model import f1, recall, precision
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from train.generator import Generator


class ModelEvaluator(object):
    def __init__(self, model_path, input_shape, color_mode):
        self._input_shape = input_shape
        self._color_mode = color_mode
        self._model = load_model(model_path,
                                 custom_objects={'f1': f1,
                                                 'recall': recall,
                                                 'precision': precision})

        self._idx_to_class = {0: 'Cell',
                              1: 'Cell-Multi',
                              2: 'Cracking',
                              3: 'Diode',
                              4: 'Diode-Multi',
                              5: 'Hot-Spot',
                              6: 'Hot-Spot-Multi',
                              7: 'No-Anomaly',
                              8: 'Offline-Module',
                              9: 'Shadowing',
                              10: 'Soiling',
                              11: 'Vegetation'}

    def predict(self, input_file):
        im = load_img(input_file, color_mode=self._color_mode)
        arr = img_to_array(im).reshape(self._input_shape)
        idx = np.argmax(self._model.predict(arr), axis=-1)
        return self._idx_to_class[idx[0]]


if __name__ == '__main__':
    evaluator = ModelEvaluator('../app/model.h5', (1, 40, 32, 3), 'rgb')
    print(evaluator.predict('../images-resized-split/train/Hot-Spot/6730.jpg'))
