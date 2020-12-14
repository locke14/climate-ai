import shutil

from os import listdir, makedirs
from os.path import isfile, join
import cv2


class Resizer(object):
    def __init__(self, data_path, input_dir, output_dir, output_size):
        self._input_path = join(data_path, input_dir)
        self._output_path = join(data_path, output_dir)
        self._output_size = output_size

    def create_directories(self):
        shutil.rmtree(self._output_path, True)
        makedirs(self._output_path)

    def resize_image(self, input_image, output_image):
        im = cv2.imread(input_image)
        top = int((self._output_size[0] - im.shape[0]) / 2)
        bottom = self._output_size[0] - top - im.shape[0]
        left = int((self._output_size[1] - im.shape[1]) / 2)
        right = self._output_size[1] - left - im.shape[1]

        im_padded = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
        cv2.imwrite(output_image, im_padded)

    def resize(self):
        self.create_directories()

        for f in [f for f in listdir(self._input_path) if isfile(join(self._input_path, f))]:
            print(f'Resizing image {f}')
            self.resize_image(join(self._input_path, f), join(self._output_path, f))


if __name__ == '__main__':
    resizer = Resizer('../', 'images', 'images-resized', (40, 32, 3))
    resizer.resize()
