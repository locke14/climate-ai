from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50

from model.base_model import BaseModel


class ResNetModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def init(self, num_training_layers, nodes, dropout):
        resnet_model = ResNet50(input_shape=self._input_shape,
                                include_top=False,
                                weights='imagenet')

        for layer in resnet_model.layers[:-num_training_layers]:
            layer.trainable = False

        self._model = Sequential()

        self._model.add(resnet_model)

        self._model.add(GlobalAveragePooling2D())

        self._model.add(Dense(nodes, activation='relu'))

        self._model.add(Dropout(dropout))

        self._model.add(Dense(self._num_classes, activation='softmax'))


if __name__ == '__main__':
    model = ResNetModel((40, 32, 3), 12)
    model.init(32, 128, 0.5)
    model.compile('categorical_crossentropy', 'SGD', ['accuracy'])
