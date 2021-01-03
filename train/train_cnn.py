from model.base_model import f1, recall, precision
from model.cnn import CNNModel
from train.train_model import ModelTrainer


class CNNTrainer(ModelTrainer):
    def __init__(self, data_path, input_shape, num_classes):
        super().__init__(data_path, input_shape, num_classes)

    def init_model(self):
        self._model = CNNModel(self._input_shape, self._num_classes)
        self._model.init([64, 128, 256], 512, 3, 2, 0.3)
        self._model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy', f1, recall, precision])


if __name__ == '__main__':
    trainer = CNNTrainer('../images-split', (40, 24, 1), 12)
    trainer.train(color_mode='grayscale', epochs=200)
    trainer.plot_history(prefix='cnn', output_dir='results')
    print(trainer.evaluate(color_mode='grayscale'))
    trainer.save_model('./results/cnn.h5')
