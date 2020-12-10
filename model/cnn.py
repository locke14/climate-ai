from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from model.base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def init(self, filters=None, nodes=None, kernel_size=None, pool_size=None, dropout=None):
        self._model = Sequential()

        for f in filters:
            self._model.add(Conv2D(f, kernel_size=(kernel_size, kernel_size),
                            activation='relu', input_shape=self._input_shape))

            self._model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(2, 2)))

        self._model.add(Flatten())

        self._model.add(Dense(nodes, activation='relu'))

        self._model.add(Dropout(dropout))

        self._model.add(Dense(self._num_classes, activation='softmax'))


if __name__ == '__main__':
    model = CNNModel((40, 24, 1), 12)
    model.init([32, 64, 64], 128, 3, 2, 0.5)
    model.compile('categorical_crossentropy', 'SGD', ['precision'])
