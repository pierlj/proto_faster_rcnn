import numpy as np
import torch

from .dataset_categories import *
from .datasets_info import * 


'''
Boxes modes constant definitions.
'''
class BBoxMode_():
    def __init__(self):
        self.XYXY = 'XYXY'
        self.XYWH = 'XYWH'
        self.REL = 'REL'
        self.ABS = 'ABS'

BBoxMode = BBoxMode_()

'''
BBox object to get easy conversion between different
representation: absolute coord, image size relative coord,
XYWH, XYXY.
'''

class BBox():
    def __init__(self, x=0, y=0, w=0, h=0, difficulty=0, label=0):
        self.x = x # x_1 i.e. left coord
        self.y = y # y_1 i.e. top coord
        self.w = w
        self.h = h
        self.x_ = None # x2 i.e. right coord
        self.y_ = None # y2 i.e. bottom coord

        self.difficulty = difficulty

        self.label = label

        self.mode = BBoxMode.REL
        self.coord_mode = BBoxMode.XYWH

    '''
    Take an input line from label file of DOTA dataset and build the
    corresponding bbox instance.

    IMPORTANT: coords are in pixels
    '''

    def to_list(self):
        if self.coord_mode == BBoxMode.XYWH:
            return [self.x, self.y, self.w, self.h]
        else: 
            return [self.x, self.y, self.x_, self.y_]

    def from_DOTA_label(self, raw_label):
        categories = DatasetCategories(dota_meta)
        tokens = raw_label[:-1].split(' ')

        self.dota_coords = list(map(lambda x: float(x), tokens[:-2]))
        # print(tokens)

        self.label = categories.name_to_label(tokens[-2])
        self.difficulty = int(tokens[-1])

        self.x = min([self.dota_coords[i]
                      for i in range(len(self.dota_coords)) if i % 2 == 0])
        self.y = min([self.dota_coords[i]
                      for i in range(len(self.dota_coords)) if i % 2 == 1])

        x_ = max([self.dota_coords[i]
                  for i in range(len(self.dota_coords)) if i % 2 == 0])
        y_ = max([self.dota_coords[i]
                  for i in range(len(self.dota_coords)) if i % 2 == 1])

        self.w = x_ - self.x
        self.h = y_ - self.y

        if self.w <= 1 and self.h <= 1:
            self.mode = BBoxMode.REL
        else:
            self.mode = BBoxMode.ABS

        return self

    '''
    Take an input line from label file of COWC dataset and build the
    corresponding bbox instance.

    IMPORTANT: coords are relative to img size
    '''

    def from_COWC_label(self, raw_label):
        tokens = raw_label[:-1].split(' ')
        self.label = int(float(tokens[0]))

        self.x, self.y, self.w, self.h = list(
            map(lambda value: float(value), tokens[1:]))

        if self.w <= 1 and self.h <= 1:
            self.mode = BBoxMode.REL
        else:
            self.mode = BBoxMode.ABS

        return self
    
    def from_XVIEW_label(self, raw_label):
        tokens = raw_label[:-1].split(' ')
        try:
            self.label = list(XVIEW_CATEGORIES.values()).index(XVIEW_CATEGORIES[tokens[-1]])
            if self.label > 60:
                print(raw_label)
        except:
            return None

        self.x, self.y, self.w, self.h = list(
            map(lambda value: float(value), tokens[:-1]))

        if self.w <= 1 and self.h <= 1:
            self.mode = BBoxMode.REL
        else:
            self.mode = BBoxMode.ABS

        return self

    def change_coord_origin(self, x0, y0):
        self.x = self.x - x0
        self.y = self.y - y0
    
    def convert_to_absolute(self, img_size):
        assert self.mode == BBoxMode.REL, 'BBox coords are already absolute'
        self.mode = BBoxMode.ABS
        self.x, self.y, self.w, self.h = img_size * np.array(
            [self.x, self.y, self.w, self.h])
        
    def convert_to_relative(self, img_size):
        assert self.mode == BBoxMode.ABS, 'BBox coords are already relative to image size'
        self.mode = BBoxMode.REL
        self.x, self.y, self.w, self.h = 1 / img_size * np.array(
            [self.x, self.y, self.w, self.h])
    
    def convert_to_xyxy(self):
        assert self.coord_mode == BBoxMode.XYWH, 'BBox coords already in XYXY mode'
        self.coord_mode = BBoxMode.XYXY
        self.x_ = self.x + self.w
        self.y_ = self.y + self.h
    
    def convert_to_xywh(self):
        assert self.coord_mode == BBoxMode.XYXY, 'BBox coords already in XYWH mode'
        self.coord_mode = BBoxMode.XYWH
        self.w = self.x_ - self.x
        self.h = self.y_ - self.y

    def get_line_cxywh(self, img_size=None):
        if self.mode == BBoxMode.ABS:
            assert img_size is not None , 'Image size required to prepare dataset as original input has boxes'

        if self.mode == BBoxMode.ABS:
            self.convert_to_relative(img_size)

        line = '{} {} {} {} {}\n'.format(
            self.label ,
            self.x,
            self.y, 
            self.w,
            self.h
        )

        return line
    
    def to_tensor(self, mode=None):
        if mode is not None and mode == BBoxMode.XYWH:
            self.convert_to_xywh()
        elif mode is not None and mode == BBoxMode.XYXY:
            self.convert_to_xyxy()

        if self.coord_mode == BBoxMode.XYWH:
            return (torch.tensor([self.x, self.y, self.w, self.h]).type(torch.FloatTensor), 
                    torch.tensor(self.label ))
        else: 
            return (torch.tensor([self.x, self.y, self.x_, self.y_]).type(torch.FloatTensor),
                     torch.tensor(self.label ))


    def clip(self, x1, y1, x2, y2):
        if self.coord_mode == BBoxMode.XYWH:
            self.convert_to_xyxy()
            convert_back = True

        self.x = max(min(self.x, x2), x1)
        self.y = max(min(self.y, y2), y1)
        self.x_ = max(min(self.x_, x2), x1)
        self.y_ = max(min(self.y_, y2), y1)

        if convert_back:
            self.convert_to_xywh()
    
    def area(self):
        if self.w is not None and self.h is not None:
            return self.w * self.h
        else:
            return (self.x_ - self.x) * (self.y_ - self.y)

    def __str__(self):
        try:
            text = 'Classe: {},\nx: {}, y: {}, w: {}, h: {}\nDifficulty: {}'.format(DOTA_CATEGORIES[self.label ],
                                                                                    self.x,
                                                                                    self.y,
                                                                                    self.w,
                                                                                    self.h,
                                                                                    self.difficulty)
        except:
            text = 'Classe: {},\nx: {}, y: {}, w: {}, h: {}\nDifficulty: {}'.format(self.label ,
                                                                                    self.x,
                                                                                    self.y,
                                                                                    self.w,
                                                                                    self.h,
                                                                                    self.difficulty)
        return text
