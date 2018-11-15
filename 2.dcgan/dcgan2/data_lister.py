import glob

import abc
import os

class DataLister(abc.ABC):
    @abc.abstractclassmethod
    def list(self):
        raise Exception('Implement Datarist.list(self)')

class FileLister(DataLister):
    def __init__(self, root, ext='*', recursive=False):
        self.root = root
        self.ext = ext
        self.recursive = recursive

    def list(self):
        query = os.path.normpath(os.path.join(root, '*.' + ext))
        files = glob.glob(query, self.recursive)
        return files