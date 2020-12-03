import os
from Data.Parser import Parser


class Splitter(object):
    def __init__(self, data_path, split_ratio):
        self._images_path = os.path.join(data_path, 'images')
        self._out_path = os.path.join(data_path, 'images-split')
        self._split_ratio = split_ratio
        self._parser = Parser(data_path)

    def split(self):
        pass


if __name__ == '__main__':
    splitter = Splitter('../', 0.1)
    splitter.split()
