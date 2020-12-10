import os
from abc import abstractmethod

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class ModelTrainer(object):
    def __init__(self, data_path, input_shape, num_classes):
        self._data_path = data_path
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = None
        self._train_generator = None
        self._test_generator = None
        self._train_history = None

    @abstractmethod
    def init_model(self):
        pass

    def init_generators(self, rescale=None, validation_split=0.2):
        self._train_generator = ImageDataGenerator(rotation_range=0.0,
                                                   width_shift_range=0.0,
                                                   height_shift_range=0.0,
                                                   rescale=rescale,
                                                   shear_range=0.0,
                                                   zoom_range=0.0,
                                                   horizontal_flip=True,
                                                   fill_mode='nearest',
                                                   validation_split=validation_split)

        self._test_generator = ImageDataGenerator(rotation_range=0.0,
                                                  width_shift_range=0.0,
                                                  height_shift_range=0.0,
                                                  rescale=1./255,
                                                  shear_range=0.0,
                                                  zoom_range=0.0,
                                                  horizontal_flip=False,
                                                  fill_mode='nearest',
                                                  validation_split=0.0)

    def flow_from_train_dir(self, color_mode='rgb', batch_size=16):
        return self._train_generator.flow_from_directory(os.path.join(self._data_path, 'train'),
                                                         target_size=self._input_shape[:2],
                                                         batch_size=batch_size,
                                                         color_mode=color_mode,
                                                         class_mode='categorical',
                                                         subset='training',
                                                         shuffle=True)

    def flow_from_validation_dir(self, color_mode='rgb', batch_size=16):
        return self._train_generator.flow_from_directory(os.path.join(self._data_path, 'train'),
                                                         target_size=self._input_shape[:2],
                                                         batch_size=batch_size,
                                                         color_mode=color_mode,
                                                         class_mode='categorical',
                                                         subset='validation',
                                                         shuffle=False)

    def flow_from_test_dir(self, color_mode='rgb', batch_size=16):
        return self._test_generator.flow_from_directory(os.path.join(self._data_path, 'test'),
                                                        target_size=self._input_shape[:2],
                                                        batch_size=batch_size,
                                                        color_mode=color_mode,
                                                        class_mode='categorical')

    def train(self, rescale=None, color_mode='rgb', epochs=50):
        self.init_model()
        self.init_generators(rescale=rescale)
        self._train_history = self._model.model.fit(self.flow_from_train_dir(color_mode=color_mode),
                                                    validation_data=self.flow_from_validation_dir(color_mode=color_mode),
                                                    epochs=epochs)

    def evaluate(self, color_mode='rgb'):
        result = self._model.model.evaluate(self.flow_from_test_dir(color_mode=color_mode))
        for i, m in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1']):
            print(f'{m}: {result[i]}')

    def predict(self, color_mode='rgb'):
        return self._model.model.predict(self.flow_from_test_dir(color_mode=color_mode))

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
