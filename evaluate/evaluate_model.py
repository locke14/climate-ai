from tensorflow.keras.models import load_model
import numpy as np
from model.base_model import f1, recall, precision
from tensorflow.keras.preprocessing.image import load_img, img_to_array


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

    def predict_from_file(self, input_file):
        im = load_img(input_file, color_mode=self._color_mode, target_size=self._input_shape[1:3])
        arr = img_to_array(im).reshape(self._input_shape)
        arr = arr/255.
        idx = np.argmax(self._model.predict(arr), axis=-1)
        return self._idx_to_class[idx[0]]

    def predict(self, arr):
        idx = np.argmax(self._model.predict(arr), axis=-1)
        return self._idx_to_class[idx[0]]


if __name__ == '__main__':
    evaluator = ModelEvaluator('../model.h5', (1, 40, 24, 1), 'grayscale')
    print(evaluator.predict_from_file('../images-split/test/No-Anomaly/10011.jpg'))
