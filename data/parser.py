import json
import os

import pandas as pd


class Parser(object):
    def __init__(self, data_path):
        self._labels_file = os.path.join(data_path, 'module_metadata.json')

        self._data = None
        self._image_list = None
        self._labels = None

    def parse(self):
        with open(self._labels_file, 'r') as f:
            self._data = pd.DataFrame(json.load(f))

        self._image_list = [i[7:] for i in self._data.loc['image_filepath'].tolist()]
        self._labels = self._data.loc['anomaly_class'].unique()

    @property
    def data(self):
        return self._data

    @property
    def image_list(self):
        return self._image_list

    @property
    def labels(self):
        return self._labels

    @property
    def binary_labels(self):
        return ['No-Anomaly', 'Anomaly']

    def get_label_images(self, label):
        return [k for k, v in enumerate(self.data.loc['anomaly_class']) if v == label]

    def get_no_anomaly_images(self):
        return [k for k, v in enumerate(self.data.loc['anomaly_class']) if v == 'No-Anomaly']

    def get_anomaly_images(self):
        return [k for k, v in enumerate(self.data.loc['anomaly_class']) if v == 'Anomaly']


if __name__ == '__main__':
    parser = Parser('../')
    parser.parse()
    print(parser.labels)
