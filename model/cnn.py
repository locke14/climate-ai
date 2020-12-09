from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D


class CNNModel(object):
    def __init__(self, input_shape, num_classes):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = None

    @property
    def model(self):
        return self._model

    def init(self, filters, nodes, kernel_size, pool_size, dropout):
        self._model = Sequential()

        self._model.add(Conv2D(filters[0], kernel_size=(kernel_size, kernel_size),
                               activation='relu', input_shape=self._input_shape))

        self._model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(2, 2)))

        self._model.add(Conv2D(filters[1], kernel_size=(kernel_size, kernel_size),
                               activation='relu'))

        self._model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(2, 2)))

        self._model.add(Conv2D(filters[2], kernel_size=(kernel_size, kernel_size),
                        activation='relu'))

        self._model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(2, 2)))

        self._model.add(Flatten())

        self._model.add(Dense(nodes, activation='relu'))

        self._model.add(Dropout(dropout))
        self._model.add(Dense(self._num_classes, activation='softmax'))

    def summary(self):
        print(self._model.summary())

    def compile(self, loss, optimizer, metrics):
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


if __name__ == '__main__':
    model = CNNModel((40, 24, 1), 12)
    model.init([32, 64, 64], 128, 3, 2, 0.5)
    model.compile('categorical_crossentropy', 'SGD', ['accuracy'])
