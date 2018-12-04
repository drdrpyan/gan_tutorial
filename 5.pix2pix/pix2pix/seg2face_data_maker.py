import cv2
import numpy as np
import dlib

class Seg2FaceDataMaker(object):
    def __init__(self, lm_detector_param, out_size, margin=1.0):
        self._out_size = out_size
        self._margin = margin

        self._f_detector = dlib.get_frontal_face_detector()
        self._lm_detector = dlib.shape_predictor(lm_detector_param)

    #def make(self, input_image, bbox):
    #    face_img = self._extract_face(input_image, bbox)
    #    seg = self._fill_seg(face_img)
    #    seg_splitted = self._split_seg(seg)
    #    return seg_splitted

    def make(self, input_image):
        resized_input = cv2.resize(
            input_image, (self._out_size[0], self._out_size[1]))

        seg = self._fill_seg(input_image)
        seg = cv2.resize(seg, (self._out_size[0], self._out_size[1]), 
                         interpolation=cv2.INTER_NEAREST)
        #seg = (seg.astype(np.float32) / 4.0) - 1.
        return resized_input, seg

    def _extract_face(self, input_image, bbox): 
        side = int(max(bbox[2], bbox[3]) * self._margin)
        border = int(side/2)
        if input_image.shape[2] == 1:
            black = [0]
        else:
            black = [0, 0, 0]
        bg = cv2.copyMakeBorder(input_image, border, border, border, border, cv2.BORDER_CONSTANT, value=black)

        center = (int(bbox[0] + border + bbox[2]/2),
                  int(bbox[1] + border + bbox[3]/2))

        cropped = bg[bbox[1]:bbox[1]+side, bbox[0]:bbox[0]+side]

        resized = cv2.resize(cropped, (self._out_size))

        return resized

    #def make(self, input_image, bbox):
    #    if len(input_image.shape) == 4:
    #        return [self._make(input_image[i], bbox[i]) for i in range(input_image.shape[0])]
    #    else:
    #        return self._make(input_image, bbox)

    #def _fill_seg(self, input_image):
    #    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #    lm = self._lm_detector(gray, [0, 0] + input_image.shape[0:2])
    #    np_lm = [(lm.part[i].x, lm.part[i].y) for i in range(lm.num_parts)]

    #    seg = np.zeros_like(gray)
        
    #    seg = cv2.fillPoly(seg, pts=np_lm[0:17], color=1)
    #    seg = cv2.fillPoly(seg, pts=np_lm[36:42], color=2)
    #    seg = cv2.fillPoly(seg, pts=np_lm[42:48], color=3)
    #    seg = cv2.fillPoly(seg, pts=np_lm[17:22], color=4)
    #    seg = cv2.fillPoly(seg, pts=np_lm[22:27], color=5)
    #    seg = cv2.fillPoly(seg, pts=(np_lm[27], np_lim[31], np_lim[35]), color=6)
    #    seg = cv2.fillPoly(seg, pts=np_lm[48:60], color=7)
    #    seg = cv2.fillPoly(seg, pts=np_lm[60:68], color=1)

    #    return seg

    def _fill_seg(self, input_image):
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        #rect = dlib.rectangle(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        rect = self._f_detector(gray)
        if len(rect) > 0:
            lm = self._lm_detector(gray, rect[0])
            np_lm = [(lm.part(i).x, lm.part(i).y) for i in range(lm.num_parts)]

            seg = np.zeros_like(gray, np.uint8)
        
            seg = cv2.fillPoly(seg, pts=np.array([np_lm[0:17]], np.int32), color=1)
            seg = cv2.fillPoly(seg, pts=np.array([np_lm[36:42]], np.int32), color=2)
            seg = cv2.fillPoly(seg, pts=np.array([np_lm[42:48]], np.int32), color=3)
            seg = cv2.fillPoly(seg, pts=np.array([np_lm[17:22]], np.int32), color=4)
            seg = cv2.fillPoly(seg, pts=np.array([np_lm[22:27]], np.int32), color=5)
            seg = cv2.fillPoly(seg, pts=np.array([[np_lm[27], np_lm[31], np_lm[35]]], np.int32), color=6)
            seg = cv2.fillPoly(seg, pts=np.array([np_lm[48:60]], np.int32), color=7)
            seg = cv2.fillPoly(seg, pts=np.array([np_lm[60:68]], np.int32), color=8)

            return seg
        else:
            return None
    

    def _split_seg(self, seg):
        shape = seg.shape
        shape[-1] = 8
        new_seg = np.zeros(shape)

        for i in range(1,8):
            new_seg[i] = seg==i

        return new_seg