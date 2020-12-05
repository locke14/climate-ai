import os
import random

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from mdutils.mdutils import MdUtils

from data.parser import Parser


class DataAnalyzer(object):
    def __init__(self, data_path):
        self._images_path = os.path.join(data_path, 'images')
        self._parser = Parser(data_path)
        self._results = MdUtils(file_name='results', title='Overview')

        self._num_images = None
        self._counts = None
        self._image_shape = None
        self._image_shape_mean = None

    def _compute_stats(self):
        self._num_images = self._parser.data.shape[1]
        self._counts = self._parser.data.loc['anomaly_class'].value_counts().to_dict()

    def _plot_random_image(self):
        random_image_file = f'{random.randint(0, self._num_images)}.jpg'
        image = img.imread(os.path.join(self._images_path, random_image_file))
        self._image_shape = image.shape

        fig = plt.figure()
        plt.tight_layout()
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.title('Random Image')
        plt.savefig('random_image.png')
        plt.close(fig)

        plt.figure()
        plt.tight_layout()
        plt.hist(image.flatten())
        plt.xlabel('Pixel Value')
        plt.ylabel('Counts')
        plt.title('Histogram')
        plt.savefig('random_image_histogram.png')
        plt.close(fig)

    def _compute_mean_shape(self):
        h, w = [], []
        for im in self._parser.image_list:
            print(f'Reading image: {im}')
            image = img.imread(os.path.join(self._images_path, im))
            h.append(image.shape[0])
            w.append(image.shape[1])
        self._image_shape_mean = (np.mean(h), np.mean(w))

    def _plot_image_each_class(self):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        plt.title('Random Image In Each Class')
        i = 1
        for label in self._parser.labels:
            image_list = self._parser.get_label_images(label)
            random_selection = random.choice(image_list)
            image_path = self._parser.data.loc['image_filepath'].tolist()[random_selection]
            image = img.imread(os.path.join(self._images_path, image_path[7:]))
            plt.subplot(4, 3, i)
            i += 1
            plt.subplots_adjust(hspace=1, wspace=1)
            plt.title(f'Class: {label}')
            cur_axes = plt.gca()
            cur_axes.axes.get_xaxis().set_ticks([])
            cur_axes.axes.get_yaxis().set_ticks([])
            plt.imshow(np.uint8(image))

        plt.savefig('random_image_each_class.png')
        plt.close(fig)

    def analyze(self):
        self._parser.parse()
        self._compute_stats()
        self._plot_random_image()
        self._compute_mean_shape()
        self._plot_image_each_class()

    def save_results(self):
        self._results.new_paragraph(f'Number of images: {self._num_images}')
        self._results.new_paragraph(f'Number of unique classes: {len(self._parser.labels)}')
        self._results.new_paragraph(f'Class names:')
        self._results.new_list(items=self._parser.labels)
        self._results.new_paragraph(f'Number of images per class: ')
        self._results.new_list(items=[f'{k}: {v}' for k, v in self._counts.items()])
        self._results.new_paragraph(f'Image shape: {self._image_shape}')
        self._results.new_paragraph(f'Mean Image shape: {self._image_shape_mean}')
        self._results.new_paragraph(self._results.new_inline_image(text='Random Image', path='random_image.png'))
        self._results.new_paragraph(self._results.new_inline_image(text='Histogram', path='random_image_histogram.png'))
        self._results.new_paragraph(self._results.new_inline_image(text='Classes', path='random_image_each_class.png'))
        self._results.create_md_file()


if __name__ == '__main__':
    analyzer = DataAnalyzer('../')
    analyzer.analyze()
    analyzer.save_results()
