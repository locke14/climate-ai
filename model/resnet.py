from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications import ResNet50


class ResNetModel(object):
    def __init__(self, input_shape, num_classes):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = None

    @property
    def model(self):
        return self._model

    def init(self, num_training_layers, nodes, dropout):
        resnet_model = ResNet50(input_shape=self._input_shape, include_top=False, weights='imagenet')

        for layer in resnet_model.layers[:-num_training_layers]:
            layer.trainable = False

        self._model = Sequential()

        self._model.add(resnet_model)

        self._model.add(Dense(nodes, activation='relu'))

        self._model.add(Dropout(dropout))

        self._model.add(Dense(self._num_classes, activation='softmax'))

    def summary(self):
        print(self._model.summary())

    def compile(self, loss, optimizer, metrics):
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


if __name__ == '__main__':
    model = ResNetModel((40, 32, 3), 12)
    model.init(32, 128, 0.5)
    model.compile('binary_crossentropy', 'SGD', ['accuracy'])