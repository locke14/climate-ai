import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Generator(object):
    def __init__(self, data_path, input_shape):
        self._data_path = data_path
        self._input_shape = input_shape
        self._train_generator = None
        self._test_generator = None

    def init(self):
        self._train_generator = ImageDataGenerator(rescale=1/255.,
                                                   horizontal_flip=True,
                                                   vertical_flip=True,
                                                   )

        self._test_generator = ImageDataGenerator(rescale=1/255.,)

    def flow_from_train_dir(self, color_mode='rgb', batch_size=32):
        return self._train_generator.flow_from_directory(os.path.join(self._data_path, 'train'),
                                                         target_size=self._input_shape[:2],
                                                         batch_size=batch_size,
                                                         color_mode=color_mode,
                                                         class_mode='categorical')

    def flow_from_validation_dir(self, color_mode='rgb', batch_size=32):
        return self._train_generator.flow_from_directory(os.path.join(self._data_path, 'train'),
                                                         target_size=self._input_shape[:2],
                                                         batch_size=batch_size,
                                                         color_mode=color_mode,
                                                         class_mode='categorical',
                                                         subset='validation',
                                                         shuffle=False)

    def flow_from_test_dir(self, color_mode='rgb', batch_size=32):
        return self._test_generator.flow_from_directory(os.path.join(self._data_path, 'test'),
                                                        target_size=self._input_shape[:2],
                                                        batch_size=batch_size,
                                                        color_mode=color_mode,
                                                        class_mode='categorical')
