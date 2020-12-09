import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.resnet import ResNetModel


class ResNetTrainer(object):
    def __init__(self, data_path, input_shape, num_classes):
        self._data_path = data_path
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = None
        self._train_generator = None
        self._test_generator = None

    def init_model(self):
        self._model = ResNetModel(self._input_shape, self._num_classes)
        self._model.init(32, 128, 0.5)
        self._model.compile('binary_crossentropy', 'SGD', ['accuracy'])

    def init_generators(self, validation_split=0.2):
        self._train_generator = ImageDataGenerator(rotation_range=0.0,
                                                   width_shift_range=0.0,
                                                   height_shift_range=0.0,
                                                   rescale=1./255,
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

    def flow_from_train_dir(self, batch_size=32):
        return self._train_generator.flow_from_directory(os.path.join(self._data_path, 'train'),
                                                         target_size=self._input_shape[:2],
                                                         batch_size=batch_size,
                                                         color_mode='rgb',
                                                         class_mode='categorical',
                                                         subset='training',
                                                         shuffle=True)

    def flow_from_validation_dir(self, batch_size=32):
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

    def train(self, epochs=20):
        self.init_model()
        self.init_generators()
        self._model.model.fit(self.flow_from_train_dir(),
                              validation_data=self.flow_from_validation_dir(),
                              epochs=epochs)


if __name__ == '__main__':
    trainer = ResNetTrainer('../images-split', (40, 32, 1), 12)
    trainer.train()
