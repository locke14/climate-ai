from tensorflow.keras.models import load_model
import numpy as np
from model.base_model import f1, recall, precision
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from train.generator import Generator


class ModelEvaluator(object):
    def __init__(self, model_path, data_path, input_shape, color_mode):
        self._input_shape = input_shape
        self._color_mode = color_mode
        self._model = load_model(model_path,
                                 custom_objects={'f1': f1,
                                                 'recall': recall,
                                                 'precision': precision})

        generator = Generator(data_path, self._input_shape)
        generator.init()
        class_to_idx = generator.flow_from_train_dir(color_mode=self._color_mode).class_indices
        self._idx_to_class = {v: k for k, v in class_to_idx.items()}

    def predict(self, input_file):
        im = load_img(input_file, color_mode=self._color_mode)
        arr = img_to_array(im).reshape(self._input_shape)
        idx = np.argmax(self._model.predict(arr), axis=-1)
        return self._idx_to_class[idx[0]]


if __name__ == '__main__':
    evaluator = ModelEvaluator('../train/results/cnn.h5', '../images-split', (1, 40, 24, 1), 'grayscale')
    print(evaluator.predict('../images/1000.jpg'))
