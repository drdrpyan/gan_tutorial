import numpy as np

import math

#import abc

#class SampleImageMaker(abc.ABC):
#    @abc.abstractclassmethod
#    def create(**ksargs):
#        raise Exception("Implement 'create()'")

#class TileImageMaker(SampleImageMaker):
#    def create(**ksargs):
#        pass

class TileImageMaker(object):
    def create(self, images, rows=None, cols=1):
        num_images = len(images)

        if num_images == 0:
            raise Exception("'images' is empty")

        self._check_image_size(images)

        r, c = self._calc_grid(num_images, rows, cols)

        if num_images > r * c:
            raise Exception("len(images) is larger than tile grids")

        bg = self._create_background(images[0].shape, r, c)

        y_begin = range(0, bg.shape[0], images[0].shape[0])
        x_begin = range(0, bg.shape[1], images[0].shape[1])
        for i in range(num_images):
            y = y_begin[i // c]
            x = x_begin[i % c]
            bg[y:y+images[i].shape[0], x:x+images[i].shape[1], :] = images[i]

        return bg


    def _check_image_size(self, images):
        shape = images[0].shape
        for img in images[1:]:
            if shape != img.shape:
                raise Exception('All images must have same shape')

    def _calc_grid(self, num_images, rows, cols):
        if (rows is None) and (cols is None):
            raise Exception("At leas one of 'rows' and 'cols' must be an integer")
        elif rows is None:
            r = math.ceil(num_images / cols)
            c = cols
        elif cols is None:
            r = rows
            c = math.ceil(num_images / rows)
        else:
            r = rows
            c = cols

        return int(r), int(c)

    def _create_background(self, img_shape, row, col):
        height = img_shape[0] * row
        width = img_shape[1] * col
        bg = np.zeros([height, width, img_shape[2]])
        return bg
        
        