import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from mdutils.mdutils import MdUtils
from model.resnet import ResNetModel
from tensorflow.keras import backend as k


def recall(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + k.epsilon())


def precision(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + k.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + k.epsilon()))


class ResNetTrainer(object):
    def __init__(self, data_path, input_shape, num_classes):
        self._data_path = data_path
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = None
        self._train_generator = None
        self._test_generator = None
        self._train_history = None

    def init_model(self):
        self._model = ResNetModel(self._input_shape, self._num_classes)
        self._model.init(32, 256, 0.1)
        self._model.compile('categorical_crossentropy', 'adam', ['accuracy', f1, recall, precision])

    def init_generators(self, validation_split=0.2):
        self._train_generator = ImageDataGenerator(rotation_range=0.0,
                                                   width_shift_range=0.0,
                                                   height_shift_range=0.0,
                                                   # rescale=1./255,
                                                   shear_range=0.0,
                                                   zoom_range=0.0,
                                                   horizontal_flip=True,
                                                   fill_mode='nearest',
                                                   validation_split=validation_split)

        self._test_generator = ImageDataGenerator(rotation_range=0.0,
                                                  width_shift_range=0.0,
                                                  height_shift_range=0.0,
                                                  # rescale=1./255,
                                                  shear_range=0.0,
                                                  zoom_range=0.0,
                                                  horizontal_flip=False,
                                                  fill_mode='nearest',
                                                  validation_split=0.0)

    def flow_from_train_dir(self, batch_size=16):
        return self._train_generator.flow_from_directory(os.path.join(self._data_path, 'train'),
                                                         target_size=self._input_shape[:2],
                                                         batch_size=batch_size,
                                                         color_mode='rgb',
                                                         class_mode='categorical',
                                                         subset='training',
                                                         shuffle=True)

    def flow_from_validation_dir(self, batch_size=16):
        return self._train_generator.flow_from_directory(os.path.join(self._data_path, 'train'),
                                                         target_size=self._input_shape[:2],
                                                         batch_size=batch_size,
                                                         color_mode='rgb',
                                                         class_mode='categorical',
                                                         subset='validation',
                                                         shuffle=False)

    def flow_from_test_dir(self, batch_size=32):
        return self._test_generator.flow_from_directory(os.path.join(self._data_path, 'test'),
                                                        target_size=self._input_shape[:2],
                                                        batch_size=batch_size,
                                                        color_mode='rgb',
                                                        class_mode='categorical')

    def train(self, epochs=25):
        self.init_model()
        self.init_generators()
        self._train_history = self._model.model.fit(self.flow_from_train_dir(),
                                                    validation_data=self.flow_from_validation_dir(),
                                                    epochs=epochs)

    def plot_history(self):
        fig = plt.figure()
        legend = []
        for m in ['accuracy', 'precision', 'recall', 'f1']:
            plt.plot(self._train_history.history[m])
            plt.plot(self._train_history.history[f'val_{m}'])
            legend.append(f'train: {m}')
            legend.append(f'test: {m}')

        plt.title('Model Metrics')
        plt.xlabel('Epoch')
        plt.legend(legend, loc='upper left')
        plt.savefig('resnet_metrics.png')
        plt.close(fig)

        fig = plt.figure()
        plt.plot(self._train_history.history['loss'])
        plt.plot(self._train_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('resnet_loss.png')
        plt.close(fig)


if __name__ == '__main__':
    trainer = ResNetTrainer('../images-split', (40, 32, 3), 12)
    trainer.train()
    trainer.plot_history()
