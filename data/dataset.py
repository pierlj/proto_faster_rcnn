import cv2
import torch
import os
import numpy as np
import random
import kornia
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image

from .utils import *
from .bbox import BBox, BBoxMode
from .augmentation import DetectionTransform, BYOL_transform
from .dataset_categories import DatasetCategories
from ..models.prototypes_utils import mask_img


class ObjectDetectionDataset(Dataset):
    '''
    Wrapper to load images and labels as pytorch tensors.
    - name: dataset name
    - root_path: path to dataset
    - transform: pytorch transform function used to pre process images
    - img_size: image are resized to this value
    - train_val_ratio: amount of data for train loader 
    - bbox_modes: convert bbox to these modes before returning labels.
    '''
    def __init__(self,  config, 
                        transform=None, 
                        train_val_ratio=0.95, 
                        bbox_modes=[BBoxMode.XYWH, BBoxMode.REL], 
                        eval_mode=False, 
                        files_paths=None,
                        target_allowed=None,
                        support_set=False,
                        all_classes_train=None,
                        background_class=False):

        self.name = config.DATASET_META.name
        self.root_path = config.DATASET_META.path
        self.config = config
        self.train_val_ratio = train_val_ratio
        self.bbox_modes = bbox_modes

        self.eval_mode = eval_mode

        self.images_path = os.path.join(self.root_path, 'images')
        self.labels_path = os.path.join(self.root_path, 'labelTxt')

        self.target_allowed = target_allowed
        self.support_set = support_set
        self.all_classes_train = all_classes_train
        self.background_class = background_class

        if files_paths is None:
            self.files_paths = []
            self.find_files_path()
        else: 
            self.files_paths = files_paths

        self.augmentation = DetectionTransform(config=self.config, 
                                                h_flip=config.DA_HFLIP, 
                                                v_flip=config.DA_VFLIP, 
                                                color=config.DA_COLOR, 
                                                affine=config.DA_AFFINE)

        if transform is not None:
            self.transform = transform
        else:
            # self.transform = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.config.DATASET_META.mean,
                                                                      std=self.config.DATASET_META.std)])
        


    def find_files_path(self):
        '''
        Assuming data is distributed in one folder or an arbitrary number of sub-folder with 
        depth 1 from root_path/images/ .
        '''

        if list_dir_only(self.images_path) != []:
            paths_to_explore = list(
                map(lambda dir: dir, os.listdir(self.images_path)))
        else:
            paths_to_explore = [""]
        
        for path in paths_to_explore:
            self.files_paths = self.files_paths + \
                list(map(lambda file: (os.path.join(self.images_path, path, file), 
                                       os.path.join(self.labels_path, path, change_file_name_extension(file, 'txt'))),
                         filter_images(os.listdir(os.path.join(self.images_path, path)))))

    def __len__(self):
        return len(self.files_paths)
    
    def __getitem__(self, idx):
        np_image = cv2.imread(self.files_paths[idx][0])
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB) / 255.0
        # np_image = np_image / 255.0
        np_image = np_image.astype(np.float32)

        if self.name == 'DIOR' and (np_image.shape[0] != 800 or np_image.shape[1] != 800):
            print(self.files_paths[idx][0])
            np_image = cv2.resize(src=np_image, dsize=(800, 800),
                         fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
        elif self.name == 'VHR':
            h, w = np_image.shape[:2]
            r_h, r_w = h / self.config.DATASET_META.max_size, w / self.config.DATASET_META.max_size
            r = min(r_h, r_w)
            np_image = cv2.resize(src=np_image, dsize=(int(h/r), int(w/r)),
                         fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

        label_file = open(self.files_paths[idx][1])
        labels_lines = label_file.readlines()
        label_file.close()        

        bbox_truth = [BBox().from_COWC_label(line)
                    for line in labels_lines]

        # BBox mode conversion according to what is setup in parameters
        if bbox_truth[0].mode == BBoxMode.ABS and BBoxMode.REL in self.bbox_modes:
            list(map(lambda bbox: bbox.convert_to_relative(np.tile(np_image.shape[:2][::-1], 2)),
                        bbox_truth))
        elif bbox_truth[0].mode == BBoxMode.REL and BBoxMode.ABS in self.bbox_modes:
            list(map(lambda bbox: bbox.convert_to_absolute(np.tile(np_image.shape[:2][::-1], 2)),
                                  bbox_truth))

        if BBoxMode.XYXY in self.bbox_modes:
            list(map(lambda bbox: bbox.convert_to_xyxy(), bbox_truth))

        np_image = self.resize(np_image)
        
        image_tensor = self.transform(np_image)

        boxes_tensor, labels_tensor = convert_box_list_to_tensor(bbox_truth)
        
        if not self.eval_mode:
            image_tensor, boxes_tensor, labels_tensor = self.augmentation(
                image_tensor, boxes_tensor, labels_tensor)

        keep_tensors = {'all_labels': None, 'task_labels': None}
        # if all_classes_train is not None then create a tensor that contains indices of 
        # annotations that belong to these classes. 
        if self.all_classes_train is not None and not self.support_set:
            classes_not_in_task = get_sets_difference(self.all_classes_train, self.target_allowed)
            allowed_labels = labels_tensor.unsqueeze(
                    -1) == classes_not_in_task
            keep_classes = torch.nonzero((allowed_labels).sum(dim=-1), as_tuple=False).view(-1)
            keep_tensors['all_labels'] = keep_classes
        if self.target_allowed is not None:
            if self.support_set:
                allowed_labels = labels_tensor.unsqueeze(-1) == self.target_allowed[idx]
            else:
                allowed_labels = labels_tensor.unsqueeze(
                    -1) == self.target_allowed
            keep = torch.nonzero((allowed_labels).sum(dim=-1), as_tuple=False).view(-1)
            keep_tensors['task_labels'] = keep


        if self.support_set:
            labels_tensor = labels_tensor[keep_tensors['task_labels']]
            boxes_tensor = boxes_tensor[keep_tensors['task_labels']]
            keep_tensors['task_labels'] = None
            
            idx_box = random.randint(0,boxes_tensor.shape[0]-1)
            # image_tensor = mask_img(image_tensor, boxes_tensor[idx_box], 0)
            return (image_tensor, boxes_tensor[idx_box:(idx_box+1), :], 
                    labels_tensor[idx_box:(idx_box+1)], keep_tensors, idx)

        if self.background_class:
            labels_tensor = labels_tensor + 1
        return image_tensor, boxes_tensor, labels_tensor, keep_tensors, idx
    
    def collate_fn(self, batch):
        img, box_tensor_list, labels_tensor_list, keep_list, indices = list(zip(*batch))
        image_tensor = torch.stack(img)
        return image_tensor, box_tensor_list, labels_tensor_list, keep_list, indices
    
    
    def get_dataloaders(self, batch_size=16, shuffle=True, num_workers=0):
        train_set_length = int(self.train_val_ratio * len(self))
        validation_set_length = len(self) - train_set_length

        
        train_set, val_set = random_split(self, lengths=[train_set_length, 
                                                        validation_set_length])
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=self.collate_fn,
                                  pin_memory=True)

        if self.train_val_ratio != 1.0:
            val_loader = DataLoader(val_set,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    collate_fn=self.collate_fn,
                                    pin_memory=True)

            return train_loader, val_loader
        else:
            return train_loader
    
    def resize(self, image):
        img_shape = image.shape[::-1][1:]
        min_size = min(img_shape)
        max_size = max(img_shape)
        ratio = min_size / self.config.DATASET_META.min_size
        if max_size * ratio > self.config.DATASET_META.max_size:
            ratio = max_size / self.config.DATASET_META.max_size

        img_shape = tuple((np.array(img_shape) / ratio).astype(int))

        image = cv2.resize(src=image, dsize=img_shape,
                         fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
        return image

class SimCLRDataset(ObjectDetectionDataset):
    def __init__(self, config, *args, **kargs):
        super().__init__(config, *args, **kargs)

        img_size = config.SSL_IMG_SIZE

        self.ssl_augment = BYOL_transform(img_size, normalize=[config.DATASET_META.mean, 
                                                                config.DATASET_META.std])

        self.cropper = transforms.Compose([transforms.RandomCrop(size=(img_size,img_size))])
    
    def __getitem__(self, idx):
        # np_image = cv2.imread(self.files_paths[idx][0])
        # np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB) / 255.0
        # np_image = np_image.astype(np.float32)
        np_image = Image.open(self.files_paths[idx][0])

        label_file = open(self.files_paths[idx][1])
        labels_lines = label_file.readlines()
        label_file.close()        

        bbox_truth = [BBox().from_COWC_label(line)
                    for line in labels_lines]

        # BBox mode conversion according to what is setup in parameters
        if bbox_truth[0].mode == BBoxMode.ABS and BBoxMode.REL in self.bbox_modes:
            list(map(lambda bbox: bbox.convert_to_relative(np.tile(np_image.shape[:2][::-1], 2)),
                        bbox_truth))
        elif bbox_truth[0].mode == BBoxMode.REL and BBoxMode.ABS in self.bbox_modes:
            list(map(lambda bbox: bbox.convert_to_absolute(np.tile(np_image.shape[:2][::-1], 2)),
                                  bbox_truth))

        if BBoxMode.XYXY in self.bbox_modes:
            list(map(lambda bbox: bbox.convert_to_xyxy(), bbox_truth))


        if self.config.SSL_PRETRAIN:
            image_tensor = self.cropper(np_image)
            labels = torch.ones(1)
            boxes = torch.ones(1,4)
            return torch.stack(self.ssl_augment(image_tensor)), boxes, labels, idx

        if self.config.SSL_RANDOM_BOXES:
            boxes_tensor = self.generate_boxes(10)
        else:
            boxes_tensor, labels_tensor = convert_box_list_to_tensor(bbox_truth)

        # boxes labels are not required for simclr
        labels_tensor = torch.arange(boxes_tensor.shape[0])
        

        image_tensor_transformed, boxes_tensor_transformed, labels_tensor_transformed = self.augmentation(
            image_tensor, boxes_tensor, labels_tensor)
        image_tensor_prime, boxes_tensor_prime, labels_tensor_prime = self.augmentation(
            image_tensor, boxes_tensor, labels_tensor)
        
        keep, keep_prime = torch.where(labels_tensor_transformed.view(-1,1) == labels_tensor_prime.view(1,-1))
        boxes_tensor_transformed, labels_tensor_transformed = boxes_tensor_transformed[keep], labels_tensor_transformed[keep]
        boxes_tensor_prime, labels_tensor_prime = boxes_tensor_prime[keep_prime], labels_tensor_prime[keep_prime]

        if boxes_tensor_prime.shape[0] == 0:
            box = self.generate_boxes(1)
            boxes_tensor_transformed = box
            boxes_tensor_prime = box
        image_tensor = torch.stack([image_tensor_transformed, image_tensor_prime], dim=0)

        # return only one box per image
        box_id = random.randint(0,boxes_tensor_prime.shape[0] - 1)
        boxes_tensor = [boxes_tensor_transformed[box_id:(box_id + 1)], boxes_tensor_prime[box_id:(box_id + 1)]]
        labels_tensor = [labels_tensor_transformed[box_id:(box_id + 1)], labels_tensor_prime[box_id:(box_id + 1)]]

        return image_tensor, boxes_tensor, labels_tensor, idx
    
    def collate_fn(self, batch):
        img, box_tensor_list, labels_tensor_list, indices = list(zip(*batch))
        image_tensor = torch.cat(img, dim=0)
        box_tensor_list = [ b for boxes in box_tensor_list for b in boxes]
        labels_tensor_list = [ l for labels in labels_tensor_list for l in labels]

        return image_tensor, box_tensor_list, labels_tensor_list, indices
    
    def generate_boxes(self, n_boxes):
        boxes = torch.rand(n_boxes, 4)
        boxes[:, :2] = boxes[:, :2] - 0.5 * boxes[:, 2:]
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        boxes = boxes.clamp(0, 1)

        return boxes