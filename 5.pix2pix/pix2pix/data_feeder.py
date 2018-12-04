#import data_lister as dl

import cv2
import numpy as np
import os
import random
import csv
     
                

class CelebABboxFeeder(object):
    def __init__(self, img_path, bbox_file, val_items=0):
        self.img_path = img_path

        bbox_f = open(bbox_file, 'r', encoding='utf-8')
        bbox_csv = csv.reader(bbox_f)
        bbox_list = list(bbox_csv)
        self._bbox_list = bbox_list[1:]
        #self._bbox_list = random.shuffle(bbox_list[1:])

        self._split_train_val(val_items)

        self._train_cursor = 0
        self._val_cursor = 0

        self.num_train = len(self._train_list)
        self.num_val = len(self._val_list)

    def get_train(self, num):
        return self._get(num, self._train_list, self._train_cursor)

    def get_validation(self, num):
        return self._get(num, self._val_list, self._val_cursor)

    def _get(self, num, l, c):
        img_list = []
        bbox_list = []
        for i in range(num):
            line = l[c]
            img_name = os.path.normpath(os.path.join(self.img_path, line[0]))
            img = cv2.imread(img_name)
            
            img_list.append(img)
            bbox_list.append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
            c = (c + 1) % len(l)

        return img_list, bbox_list

    def _split_train_val(self, val_items):
        if val_items == 0:
            self._train_list = self._bbox_list
            self._val_list = None
        elif type(val_items) == int:
            self._train_list = self._bbox_list[val_items:]
            self._val_list = self._bbox_list[:val_items]
        else:
            num_val = int(len(self._bbox_list) * val_items)
            self._train_list = self._bbox_list[num_val:]
            self._val_list = self._bbox_list[:num_val]

            
            