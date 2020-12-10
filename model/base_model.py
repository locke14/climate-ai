from abc import abstractmethod

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


class BaseModel(object):
    def __init__(self, input_shape, num_classes):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._model = None

    @property
    def model(self):
        return self._model

    @abstractmethod
    def init(self, **kwargs):
        pass

    def summary(self):
        print(self._model.summary())

    def compile(self, loss=None, optimizer=None, metrics=None):
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
