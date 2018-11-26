import cv2
import numpy as np
import dlib

class Seg2FaceDataMaker(object):
    def __init__(self, lm_detector_param, out_size, margin=1.0):
        self.out_size = out_size
        self.margin = margin

        self.lm_detector = dlib.shape_predictor(lm_detector_param)

    def make(self, input_image, bbox):
        for i in 

    def _extract_face(self, input_image, bbox):
        new_size = bbox[2:] * margin
        lr_margin = (new_size[2] - bbox[2]) / 2.0
        tb_margin = (new_size[3] - bbox[3]) / 2.0

        if input_image.shape[2] == 1:
            black = [0]
        else:
            black = [0, 0, 0]
        bg = cv2.copyMakeBorder(input_image, tb_margin, tb_margin, lr_margin, lr_margin, cv2.BORDER_CONSTANT, value=black)
        cropped = bg[bbox[1]:bbox[1]+new_size[1], bbox[0]:bbox[0]+new_size[0]]

        resized = cv2.resize(cropp2d, (self.out_size))

        return resized

    #def make(self, input_image, bbox):
    #    if len(input_image.shape) == 4:
    #        return [self._make(input_image[i], bbox[i]) for i in range(input_image.shape[0])]
    #    else:
    #        return self._make(input_image, bbox)

    def _fill_seg(self, input_image):
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        lm = lm_detector(gray, [0, 0] + input_image.shape[0:2])
        np_lm = [(lm.part[i].x, lm.part[i].y) for i in range(lm.num_parts)]

        seg = np.zeros_like(gray)
        
        seg = cv2.fillPoly(seg, pts=np_lm[0:17], color=1)
        seg = cv2.fillPoly(seg, pts=np_lm[36:42], color=2)
        seg = cv2.fillPoly(seg, pts=np_lm[42:48], color=3)
        seg = cv2.fillPoly(seg, pts=np_lm[17:22], color=4)
        seg = cv2.fillPoly(seg, pts=np_lm[22:27], color=5)
        seg = cv2.fillPoly(seg, pts=(np_lm[27], np_lim[31], np_lim[35]), color=6)
        seg = cv2.fillPoly(seg, pts=np_lm[48:60], color=7)
        seg = cv2.fillPoly(seg, pts=np_lm[60:68], color=1)

        return seg

    def _split_seg(self, seg):
        shape = seg.shape
        shape[-1] = 8
        new_seg = np.zeros(shape)

        for i in range(1,8):
            new_seg[i] = seg==i

        return new_seg
        