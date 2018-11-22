#import data_lister as dl

import numpy as np

import abc
import threading
import queue

class BaseDataFeeder(abc.ABC):
    @abc.abstractclassmethod
    def get(self, num):
        raise Exception("Implement 'get(self, num)'")

class NoiseFeeder(BaseDataFeeder):
    def __init__(self, shape):
        self.shape = shape

    def get(self, num):
        return self.get2(num, self.shape, -1, 1)

    def get2(self, num, shape, low, high):
        #noise = [np.random.uniform(low, high, shape) for i in range(num)]
        noise = np.random.uniform(low, high, size=([num]+shape))
        return noise

class MNISTFeeder(BaseDataFeeder):
    def __init__(self, data_pass, label_pass=None, train=True, one_hot=True):
        self._DATA_HEAD = 16
        self._LABEL_HEAD = 8

        self._DATA_SIZE = 784
        self._LABEL_SIZE = 1

        self._DATA_SHAPE = [28, 28, 1]
        self._LABEL_SHAPE = [1, 1, 1]
        self._ONE_HOT_LABEL_SHAPE = [1, 1, 10]

        self._NUM_TRAIN_DATA = 60000
        self._NUM_TEST_DATA = 10000

        self._NUM_CLASS = 10

        self._ONE_HOT = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

        self.data_stream = open(data_pass, 'rb')
        self.data_stream.read(self._DATA_HEAD)

        if label_pass is not None:
            self.label_stream = open(label_pass, 'rb')
            self.label_stream.read(self._LABEL_HEAD)
        else:
            self.label_stream = None

        if train:
            self.num_data = self._NUM_TRAIN_DATA
        else:
            self.num_data = self._NUM_TEST_DATA

        self.data_shape = self._DATA_SHAPE
        if one_hot:
            self.label_shape = self._ONE_HOT_LABEL_SHAPE
        else:
            self.label_shape = self._LABEL_SHAPE
        self.shape = [self.data_shape, self.label_shape]

        self.num_class = self._NUM_CLASS

        self.one_hot = one_hot

    def get(self, num):
        if self.label_stream is not None:
            return self._get_data(num), self._get_label(num)
        else:
            return self._get_data(num)

    def rewind(self):
        self.data_stream.seek(self._DATA_HEAD)
        if self.label_stream is not None:
            self.label_stream.seek(self._LABEL_HEAD)

    def _get_data(self, num):
        data_list = []
        while len(data_list) != num:
            data = self.data_stream.read(self._DATA_SIZE)
            if not data:
                self.data_stream.seek(self._DATA_HEAD)
            else:
                data_mat = self._bytes_to_mat(data)
                data_mat = data_mat.astype(np.float32) / 255
                data_list.append(data_mat)

        return data_list

    def _get_label(self, num):
        label_list = []
        while len(label_list) != num:
            label = self.label_stream.read(self._LABEL_SIZE)
            if not label:
                self.label_stream.seek(self._LABEL_HEAD)
            else:
                label = label[0]
                if self.one_hot:
                    label = self._byte_to_one_hot(label)
                else:
                    label = np.float32(label)

                label_list.append(label)

        return label_list

    def _bytes_to_mat(self, b):
        mat1d = [i for i in b]
        mat = np.reshape(mat1d, self._DATA_SHAPE)
        return mat
        
    def _byte_to_one_hot(self, b):
        return self._ONE_HOT[b]
                
                

