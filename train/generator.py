import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Generator(object):
    def __init__(self, data_path, input_shape):
        self._data_path = data_path
        self._input_shape = input_shape
        self._train_generator = None
        self._test_generator = None

    def init(self, rescale=None, validation_split=0.2):
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
