import os
from abc import abstractmethod

import matplotlib.pyplot as plt

from train.generator import Generator


class ModelTrainer(object):
    def __init__(self, data_path, input_shape, num_classes):
        self._data_path = data_path
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = None
        self._train_history = None
        self._generator = Generator(self._data_path, self._input_shape)

    @abstractmethod
    def init_model(self):
        pass

    def train(self, color_mode='grayscale', epochs=50):
        self.init_model()
        self._generator.init()
        self._train_history = self._model.model.fit(self._generator.flow_from_train_dir(color_mode=color_mode),
                                                    validation_data=self._generator.flow_from_test_dir(color_mode=color_mode),
                                                    epochs=epochs)

    def save_model(self, out_file):
        return self._model.save(out_file)

    def evaluate(self, color_mode='rgb'):
        result = self._model.model.evaluate(self._generator.flow_from_test_dir(color_mode=color_mode))
        for i, m in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1']):
            print(f'{m}: {result[i]}')

    def predict(self, color_mode='rgb'):
        return self._model.model.predict(self._generator.flow_from_test_dir(color_mode=color_mode))

    def plot_history(self, prefix='', output_dir=None):
        if output_dir:
            output_dir = os.path.join('./', output_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        else:
            output_dir = './'

        for m in ['accuracy', 'precision', 'recall', 'f1']:
            if m in self._train_history.history:
                fig = plt.figure()
                plt.plot(self._train_history.history[m])
                plt.plot(self._train_history.history[f'val_{m}'])
                plt.title(f'{m}')
                plt.xlabel('Epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(os.path.join(output_dir, f'{prefix}_{m}.png'))
                plt.close(fig)

        fig = plt.figure()
        plt.plot(self._train_history.history['loss'])
        plt.plot(self._train_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(output_dir, f'{prefix}_loss.png'))
        plt.close(fig)
