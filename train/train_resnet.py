from model.resnet import ResNetModel
from model.base_model import f1, recall, precision

from train.train_model import ModelTrainer


class ResNetTrainer(ModelTrainer):
    def __init__(self, data_path, input_shape, num_classes):
        super().__init__(data_path, input_shape, num_classes)

    def init_model(self):
        self._model = ResNetModel(self._input_shape, self._num_classes)
        self._model.init(32, 128, 0.5)
        self._model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy', f1, recall, precision])


if __name__ == '__main__':
    trainer = ResNetTrainer('../images-resized-split-binary', (40, 32, 3), 2)
    trainer.train(rescale=None, color_mode='rgb', epochs=50)
    trainer.plot_history(prefix='resnet_no_augmentation_resized_binary', output_dir='results')
    print(trainer.evaluate(color_mode='rgb'))
    trainer.save_model('./results/resnet_no_augmentation_resized_binary.h5')
