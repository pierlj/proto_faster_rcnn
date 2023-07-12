import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


from .utils import *
from .bbox import BBox

'''
Wrapper for Penn Fudan dataset.
'''

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms_=None):
        self.root = root
        self.name = 'pennfudan'
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        self.transforms = transforms.Compose([
            transforms.Resize((240, 240), interpolation=0),
            transforms.ToTensor(),
            normalize
        ])
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        a = 240 / img.size[0]
        b = 240 / img.size[1]

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1]) * a
            xmax = np.max(pos[1]) * b
            ymin = np.min(pos[0]) * a 
            ymax = np.max(pos[0]) * b
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img = self.transforms(img)
        
        # Careful here bbox are in relative mode
        box_list = [BBox(x=box[0]/240, y=box[1]/240, w=(box[2] - box[0])/240, h=(box[3] - box[1])/240, classe=labels[i]) for i, box in enumerate(boxes)]

        return img, box_list, image_id
    
    def collate_fn(self, batch):
        img, box_list, indices = list(zip(*batch))
        box_tensor_list, labels_tensor_list = list(zip(*list(map(lambda box: convert_box_list_to_tensor(box),
                                                                 box_list))))
        return torch.stack(img), box_tensor_list, labels_tensor_list, indices
    

    def get_dataloader(self):
        return DataLoader(self, batch_size=4, collate_fn=self.collate_fn)

    def __len__(self):
        return len(self.imgs)
