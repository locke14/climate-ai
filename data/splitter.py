import os
import shutil

import numpy as np

from data.parser import Parser


class Splitter(object):
    def __init__(self, data_path, split_ratio):
        self._images_path = os.path.join(data_path, 'images-resized')
        self._out_path = os.path.join(data_path, 'images-resized-split')
        self._split_ratio = split_ratio
        self._parser = Parser(data_path)

    def create_directories(self, labels):
        shutil.rmtree(self._out_path, True)
        os.makedirs(self._out_path)
        for d in ['test', 'train']:
            for label in labels:
                os.makedirs(os.path.join(self._out_path, d, label))

    def get_train_test_image_list(self, image_list):
        np.random.shuffle(image_list)
        train_images, test_images = np.split(np.array(image_list),
                                             [int(len(image_list) * self._split_ratio), ])

        return train_images, test_images

    def copy_images(self, images, base_dir, label):
        for image in images:
            src = os.path.join(self._images_path, image)
            dst = os.path.join(self._out_path, base_dir, label, image)
            if os.path.isfile(src):
                print(f'Copying {src} to {dst}')
                shutil.copy(src, dst)

    def split(self):
        self._parser.parse()
        self.create_directories(self._parser.labels)

        for label in self._parser.labels:
            train_images, test_images = self.get_train_test_image_list(self._parser.get_label_images(label))
            self.copy_images(train_images, 'train', label)
            self.copy_images(test_images, 'test', label)

    def split_binary(self):
        self._parser.parse()
        self.create_directories(self._parser.binary_labels)

        train_images, test_images = self.get_train_test_image_list(self._parser.get_no_anomaly_images())
        self.copy_images(train_images, 'train', 'No-Anomaly')
        self.copy_images(test_images, 'test', 'No-Anomaly')

        train_images, test_images = self.get_train_test_image_list(self._parser.get_anomaly_images())
        self.copy_images(train_images, 'train', 'Anomaly')
        self.copy_images(test_images, 'test', 'Anomaly')


if __name__ == '__main__':
    splitter = Splitter('../', 0.9)
    splitter.split()
    # splitter.split_binary()
