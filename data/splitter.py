import os
import shutil

import numpy as np

from data.parser import Parser


class Splitter(object):
    def __init__(self, data_path, split_ratio):
        self._images_path = os.path.join(data_path, 'images')
        self._out_path = os.path.join(data_path, 'images-split')
        self._split_ratio = split_ratio
        self._parser = Parser(data_path)

    def create_directories(self):
        shutil.rmtree(self._out_path, True)
        os.makedirs(self._out_path)
        for d in ['test', 'train']:
            for label in self._parser.labels:
                os.makedirs(os.path.join(self._out_path, d, label))

    def get_train_test_image_list(self, label):
        image_list = self._parser.get_label_images(label)
        np.random.shuffle(image_list)
        train_images, test_images = np.split(np.array(image_list),
                                             [int(len(image_list) * self._split_ratio), ])

        return train_images, test_images

    def copy_images(self, images, base_dir, label):
        for image in images:
            src = os.path.join(self._images_path, f'{image}.jpg')
            dst = os.path.join(self._out_path, base_dir, label, f'{image}.jpg')
            if os.path.isfile(src):
                print(f'Copying {src} to {dst}')
                shutil.copy(src, dst)

    def split(self):
        self._parser.parse()
        self.create_directories()

        for label in self._parser.labels:
            train_images, test_images = self.get_train_test_image_list(label)
            self.copy_images(train_images, 'train', label)
            self.copy_images(test_images, 'test', label)


if __name__ == '__main__':
    splitter = Splitter('../', 0.9)
    splitter.split()
