import numpy as np
import scipy.misc
import glob
import os

class ImageLoader(object):
    def __init__(self, path, format='png', shuffle=True, resize=None, center_crop=None, color='auto', norm_range=[-1, 1]):
        self._PIXEL_MAX = 255

        self._init_resize(resize)
        self._init_center_crop(center_crop)
        self._init_image_shape(color)

        self.norm_range = norm_range

        self._init_img_list(path, format, shuffle)

    def _init_resize(self, resize):
        if resize is None:
            self.resize = None
        elif isinstance(resize, int) == 1:
            self.resize = [resize, resize]
        elif len(resize) == 2:
            self.resize = resize
        else:
            raise Exception("'resize' must be a scalar or 2-tuple")

    def _init_center_crop(self, center_crop):
        if center_crop is None:
            self.center_crop = None
        elif isinstance(center_crop, int) == 1:
            self.center_crop = [center_crop, center_crop]
        elif len(center_crop) == 2:
            self.center_crop = center_crop
        else:
            raise Exception("'center_crop' must be a scalar or 2-tuple")

    def _init_img_list(self, path, format, shuffle):
        self.img_list = glob.glob(os.path.normpath(os.path.join(path, '*.' + format)))

        self.num_images = len(self.img_list)

        if self.num_images == 0:
            raise Exception("[!] No data found in '" + path + "'")
        #if len(self.img_list) < self.batch_size:
        #    raise Exception("[!] Entire dataset size is less than the configured batch_size")

        if shuffle:
            np.random.shuffle(self.img_list)

    def _init_image_shape(self, color):
        color_upper = color.upper()
        
        if color_upper == 'GRAY':
            img_depth = 1
        elif color_upper == 'COLOR':
            img_depth = 3
        elif color_upper == 'AUTO':
            init_image = scipy.misc.imread(img_list[0])
            if len(init_image.shape) >= 3:
                img_depth = init_image.shape[-1]
            else:
                img_depth = 1

        self.image_shape = [0, 0, img_depth]
        if self.resize is not None:
            self.image_shape[0:2] = self.resize
        elif self.center_crop is not None:
            self.image_shape[0:2] = self.center_crop
        else:
            self.image_shape[0:2] = init_image.shape[0:2]
            
 
    def _read_image(self, path):
        if self.image_shape[2] == 1:
            image = scipy.misc.imread(path, flatten=True).astype(np.float32)
        else:
            image = scipy.misc.imread(path).astype(np.float32)

        if self.center_crop is not None:
            image = self._crop_center(image, self.center_crop)

        if self.resize is not None:
            image = scipy.misc.imresize(image, self.resize)

        if self.norm_range is not None:
            image = (image / (self._PIXEL_MAX / (self.norm_range[1] - self.norm_range[0]))) + self.norm_range[0]

        return image

    def _crop_center(self, img, crop_size):
        y_begin = int(round((img.shape[0] - crop_size[0]) / 2.))
        x_begin = int(round((img.shape[1] - crop_size[1]) / 2.))

        return img[y_begin:y_begin+crop_size[0], x_begin:x_begin+crop_size[1]]
    
    #def load(self, num):
    #    pass

    def load_by_idx(self, idx):
        return [self._read_image(self.img_list[i]) for i in idx]