import data_lister as dl

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
        noise = [np.random.uniform(low, high, shape) for i in range(num)]
        return noise

class DataFeeder(BaseDataFeeder):
    def __init__(self, data_lister, shuffle=True, rewind=True, rewind_shuffle=False):
        self.cursor = 0

        self.list = data_lister.list()
        if shuffle:
            np.random.shuffle(self.list)

        self.rewind = rewind
        self.rewind_shuffle = rewind_shuffle

    def no_more_exist(self):
        return self.cursor == len(self.list)


    def get(self, num):
        if num <= 0:
            raise Exception('Request one or more items!')

        if not self.rewind and (cursor + num) > len(self.list):
            raise Exception(
                'You requested %d item(s) but only %d item(s) is(are) lefted!' 
                    % (num, len(self.list) - cursor))

        num_unfetched = num
        fetched = []
        while num_unfetched > 0:
            beg = self.cursor
            end = min(beg + num_unfetched, len(self.list))
            fetched.append([self._fetch(self.list[i]) for i in range(beg, end)])
            
            num_unfetched -= (end - beg)
            self.cursor = end

            if self.no_more_exist() and self.rewind:
                self._rewind()
            

    @abc.abstractclassmethod
    def _fetch(self, target):
        raise Exception("Implement 'DataFeeder._fetch(self, target)'")

    def _rewind(self):
        self.cursor = 0
        if self.rewind_shuffle:
            np.random.shuffle(self.list)


class PrefetchDataFeeder(DataFeeder):
    def __init__(self, 
                 data_lister, shuffle=True, rewind=True, rewind_shuffle=False, 
                 prefetch_size=128):
        super().__init__(data_lister, shuffle, rewind, rewind_shuffle)

        self.prefetch_queue = queue.Queue(prefetch_size)

        self._prefetch_thread = threading.Thread(target=self._prefetch_loop)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.run()

    def _prefetch_loop(self):
        if super()._rewind:
            while True:
                self._prefetch()
        else:
            while not super().no_more_exist():
                self._prefetch()

    def _prefetch(self):
        item = super().get(1)
        self.prefetch_queue.put(item, block=True)
                

    def get(self, num):
        item = [self.prefetch_queue.get() for i in range(num)]