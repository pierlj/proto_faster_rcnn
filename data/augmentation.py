import torch
import os
import sys
import numpy as np
import random
import kornia
from torchvision import transforms
from PIL import Image, ImageOps

from .bbox import BBoxMode


class DetectionTransform():
    def __init__(self, 
                 config=None,
                 h_flip=0.5,
                 v_flip=0.5,
                 affine=0.5,
                 color=0.5,
                 blur=0.0,
                 contrast=[0.8, 1.2],
                 brightness=[0.8, 1.2],
                 saturation=[0.8, 1.2],
                 hue=[-0.5, 0.5],
                 degrees=10,
                 scale=(0.9,1.1),
                 translate=(0.05,0.05),
                 bbox_modes=[BBoxMode.XYXY, BBoxMode.REL]):
        self.config = config
        self.params = {'h_flip':h_flip, 
                       'v_flip':v_flip,
                       'affine': affine,
                       'color':color, 
                       'blur':blur}
        self.modes = bbox_modes
        self.color_augment = kornia.augmentation.ColorJitter(p=1.0,
                                                             brightness=brightness,
                                                            contrast=contrast,
                                                            saturation=saturation,
                                                            hue=hue)
        self.affine_transform = kornia.augmentation.RandomAffine(p=1.0,
                                                                degrees=degrees,
                                                                translate=translate,
                                                                scale=scale,
                                                                padding_mode='zeros')
    
    def __call__(self, images, boxes, labels):
        output_list = True
        images = self.denormalize(images)
        if not isinstance(boxes, list):
            boxes = [boxes]
            images = [images]
            labels = [labels]
            output_list = False
        images_t, boxes_t, labels_t = [], [], []
        for img, box, label in zip(images, boxes, labels):
            img, box, keep = self.structural_tranforms(img, box)  
            img = self.color_transforms(img)
            img = self.apply_blur(img)
            # img = self.normalize(img)
            assert label[keep].shape[0] == box.shape[0], 'Boxes tensor and labels tensor sizes don\'t match'
            labels_t.append(label[keep])
            images_t.append(self.normalize(img))
            boxes_t.append(box)
        return (torch.stack(images_t, dim=0), boxes_t, labels_t) if output_list else (images_t[0], boxes_t[0], labels_t[0])

    def structural_tranforms(self, img, boxes):
        keep = torch.Tensor([True for i in range(boxes.shape[0])]).type(torch.BoolTensor)
        if random.random() < self.params['h_flip']:
            img, boxes = self.h_flip(img, boxes)
        if random.random() < self.params['v_flip']:
            img, boxes = self.v_flip(img, boxes)
        if random.random() < self.params['affine']:
            img, boxes, keep = self.affine(img, boxes)
        return img, boxes, keep
    
    def color_transforms(self, img):
        if random.random() < self.params['color']:
            if random.random() < 0.5:
                img = kornia.color.rgb_to_grayscale(img)
            else:
                img = self.color_augment(img)[0]
        return img    
    
    def apply_blur(self, img):
        if random.random() < self.params['blur']:
            pass
        return img
    
    def h_flip(self, img, boxes):
        img = img.flip([-1])        
        pivot = 1 if BBoxMode.REL in self.modes else img.shape[-1]

        if BBoxMode.XYXY in self.modes:
            boxes = torch.stack(
                [pivot - boxes[:,2], boxes[:,1], pivot - boxes[:,0], boxes[:,3]], dim=1)
        else:
            boxes = torch.stack(
                [pivot - boxes[:, 0] - boxes[:, 2], boxes[:, 1], boxes[:, 2], boxes[:, 3]], dim=1)
        return img, boxes
    
    def v_flip(self, img, boxes):
        img = img.flip([-2])
        pivot = 1 if BBoxMode.REL in self.modes else img.shape[-2]

        if BBoxMode.XYXY in self.modes:
            boxes = torch.stack(
                [boxes[:,0], pivot - boxes[:,3], boxes[:,2], pivot - boxes[:,1]], dim=1)
        else:
            boxes = torch.stack(
                [boxes[:, 0], pivot - boxes[:, 1] - boxes[:, 3], boxes[:, 2], boxes[:, 3]], dim=1)
        return img, boxes
    
    def affine(self, img, boxes):

        img = self.affine_transform(img)
        mat  = self.affine_transform.transform_matrix
        R = mat[0, :2,:2]
        t = mat[0,:2,2]
        
        if BBoxMode.REL in self.modes:
            t = (t / torch.Tensor(list(img.shape[-2:])).to(t.device)).reshape(2, 1)
            
        if BBoxMode.XYWH in self.modes:
            boxes[:,2:] = boxes[:,2:] + boxes[:, :2]
        
        areas = (boxes[:,2:] - boxes[:,:2])[:,0] * (boxes[:,2:] - boxes[:,:2])[:,1]
        '''
        To properly rotate the boxes we must apply the trasnformation on each corner.
        Therefore boxes_4p = [[x1, y1, ..., x4, y4]]
        '''
        boxes_4p = torch.stack([boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,1], 
                              boxes[:,2], boxes[:,3], boxes[:,0], boxes[:,3]], dim=1)

        boxes_t = R@boxes_4p.reshape(-1,4,2).permute((0,2,1)) + t.reshape(2,1)
        boxes_t = boxes_t.permute(0,2,1).reshape(-1,8)
        boxes_t = torch.stack([boxes_t[:,::2].min(dim=1)[0], boxes_t[:,1::2].min(dim=1)[0] , 
                           boxes_t[:,::2].max(dim=1)[0], boxes_t[:,1::2].max(dim=1)[0]], dim=1)
        
        if BBoxMode.XYWH in self.modes:
            boxes_t[:,2:] = boxes_t[:,2:] - boxes_t[:, :2]
        
        areas_t = (boxes_t[:,2:] - boxes_t[:,:2])[:,0] * (boxes_t[:,2:] - boxes_t[:,:2])[:,1]
        boxes_processed = boxes_t.clamp(0,1).type(torch.float32)

        areas_processed = (boxes_processed[:,2:] - boxes_processed[:,:2])[:,0] * \
            (boxes_processed[:,2:] - boxes_processed[:,:2])[:,1]

        keep = (areas_t * 0.5 < areas_processed)
        boxes_processed = boxes_processed[keep]
        return img[0], boxes_processed, keep

    def denormalize(self, image):
        mean = torch.Tensor(self.config.DATASET_META.mean).to(image).reshape(3,1,1)
        std = torch.Tensor(self.config.DATASET_META.std).to(image).reshape(3,1,1)

        return image * std + mean
    
    def normalize(self, image):
        mean = torch.Tensor(self.config.DATASET_META.mean).to(image).reshape(3,1,1)
        std = torch.Tensor(self.config.DATASET_META.std).to(image).reshape(3,1,1)

        return (image - mean) / std 


class BYOL_transform:
    ## Code from https://github.com/PatrickHua/SimSiam/blob/HEAD/augmentations/byol_aug.py
    def __init__(self, image_size, normalize=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]):

        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0)), # simclr paper gives the kernel size. Kernel size has to be odd positive number with torchvision
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * image_size))], p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])


    def __call__(self, x):
        x1 = self.transform1(x) 
        x2 = self.transform2(x) 
        return x1, x2

class Solarization():
    # ImageFilter
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)