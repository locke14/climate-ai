from model.resnet import ResNetModel
from model.base_model import f1, recall, precision

from train.train_model import ModelTrainer


class ResNetTrainer(ModelTrainer):
    def __init__(self, data_path, input_shape, num_classes):
        super().__init__(data_path, input_shape, num_classes)

    def init_model(self):
        self._model = ResNetModel(self._input_shape, self._num_classes)
        self._model.init(32, 256, 0.1)
        self._model.compile('categorical_crossentropy', 'adam', ['accuracy', f1, recall, precision])


if __name__ == '__main__':
    trainer = ResNetTrainer('../images-split', (40, 32, 3), 12)
    trainer.train(rescale=None, color_mode='rgb', epochs=5)
    trainer.plot_history(prefix='resnet', output_dir='results')
