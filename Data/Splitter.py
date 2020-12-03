import os
import shutil

from Data.Parser import Parser


class Splitter(object):
    def __init__(self, data_path, split_ratio):
        self._images_path = os.path.join(data_path, 'images')
        self._out_path = os.path.join(data_path, 'images-split')
        self._split_ratio = split_ratio
        self._parser = Parser(data_path)

    def split(self):
        self._parser.parse()
        shutil.rmtree(self._out_path)
        os.makedirs(self._out_path)
        for d in ['test', 'train']:
            for label in self._parser.labels:
                os.makedirs(os.path.join(self._out_path, d, label))


if __name__ == '__main__':
    splitter = Splitter('../', 0.1)
    splitter.split()
