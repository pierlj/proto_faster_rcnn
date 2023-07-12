import os
import re
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from torchvision.ops.boxes import box_iou


from .bbox import BBox

from pycocotools.coco import COCO

'''
Read a label file and create bbox objects for each label in the image.
'''

def get_object_from_label_file(path, mode='DOTA'):
    file = open(path)
    lines = file.readlines()
    file.close()
    labels = []
    if mode == 'auto':
        if lines[1][:3] == 'gsd':
            mode = 'DOTA'
        else:
            mode = 'other'

    if mode == 'DOTA':
        lines = lines[2:]
    

    for line in lines:
        label = {}
        label['raw'] = line
        if mode == 'DOTA':
            label['bbox'] = BBox().from_DOTA_label(line)
        # Labels should be stored as 'x y w h class', as for the file in COWC dataset
        elif mode == 'xview':
            label['bbox'] = BBox().from_XVIEW_label(line)
        # Labels should be stored as 'class x y w h', as for the file in COWC dataset
        else:
            label['bbox'] = BBox().from_COWC_label(line)

        # when a class is unknown discard the box    
        if label['bbox'] is not None:
            labels.append(label)
    return labels


def list_dir_only(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name[0] != '.']

'''
Remove some kind of image with wrong format (keep only jpg and png)
+ Avoid taking ground truth image from some dataset.
'''
def filter_images(img_list):
    kept_images = []
    for file in img_list:
        tokens = file.split('.')
        if tokens[-1] in ['jpg', 'png'] and tokens[-2] != 'check':
            kept_images.append(file)
    return kept_images


def change_file_name_extension(name, ext_to):
    tokens = name.split('.')
    name = '.'.join(tokens[:-1]+[ext_to])
    # tokens = name.split('/')
    # tokens = ['labelTxt' if t == 'images' else t for t in tokens]
    # name = os.path.join(*tokens)
    return name.replace('images', 'labelTxt')

def convert_box_list_to_tensor(box_list):
    box_stack, label_stack = [], []

    for box in box_list:
        box_tensor, label = box.to_tensor()
        box_stack.append(box_tensor)
        label_stack.append(label)

    return torch.stack(box_stack), torch.stack(label_stack)

'''
Find out if script is run in IPython or standard python interpreter
'''
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter



def tensor_stats(tensor):
    return tensor.min(), tensor.max(), tensor.mean(), tensor.std()

def filter_annotations(boxes_list, labels_list, keep_list, mode='task_labels'):
    boxes_filtered = []
    labels_filtered = []
    for boxes, labels, keep in zip(boxes_list, labels_list, keep_list):
        keep_idx = keep[mode]
        if keep_idx is not None:
            boxes_filtered.append(boxes[keep_idx])
            labels_filtered.append(labels[keep_idx])
        else:
            boxes_filtered.append(boxes)
            labels_filtered.append(labels)
            
    return boxes_filtered, labels_filtered

def get_sets_difference(a, b):
    '''
    Compute the difference set of the element in tensor A and tensor B
    Return the difference as a tensor.
    '''
    set_a = set(a.tolist())
    set_b = set(b.tolist())

    diff = set_a.difference(set_b)
    return torch.Tensor(list(diff)).type_as(a)

def draw_img_boxes_proposals(img, boxes, proposals, dataset_meta, scores=None):
    mean = dataset_meta.mean
    std = dataset_meta.std
    idx = 0
    img_ = torch.clamp(img[idx].cpu().permute(1,2,0) * torch.tensor(std) + torch.tensor(mean), 0, 1)
    img_ = img_.numpy()

    fig,ax = plt.subplots(1, figsize=(10,10))
    colormap = cm.get_cmap('spring')
    norm = plt.Normalize(0,1)

    # Display the image
    ax.imshow(img_)


    if scores is not None:
        color = colormap(norm(scores[idx]).cpu().numpy())
    else:
        iou_proposal = box_iou(boxes[idx], proposals[idx])
        color = colormap(iou_proposal.max(dim=0)[0].cpu().numpy())
    
    # draw proposals
    for idx_boxe, box in reversed(list(enumerate(proposals[idx]))):
        x1, y1, x2, y2 = box.cpu().detach().tolist() 
        patch = patches.Rectangle((x1, y1), x2 -x1, y2 - y1, linewidth=1, edgecolor=color[idx_boxe], facecolor='none')
        if iou_proposal.max(dim=0)[0].cpu().numpy()[idx_boxe] > 0.2:
            ax.add_patch(patch)
    
    # draw gt 
    for box in boxes[idx]:
        x1, y1, x2, y2 = box.tolist()
        patch = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(patch)

    plt.show()

def box_enlarger(boxes_list, img_size=512, min_size=32.0):
    enlarged_boxes = []
    for boxes in boxes_list:
        box = boxes.clone()
        wh = box[:, 2:] - box[:, :2]
        r = torch.sqrt(min_size**2/(wh[:,0] * wh[:, 1])).view(-1,1)
        r = torch.where(r >=1, r, torch.ones_like(r))
        box[:, :2] = box[:, :2] - 0.5 * (r - 1) * wh
        box[:, 2:] = box[:, 2:] + 0.5 * (r - 1) * wh
        box.clamp(0,img_size)
        enlarged_boxes.append(box)
    return enlarged_boxes


def build_sections_from_file(path):
    pattern = re.compile("###\s[A-Z].*")
    indices_section = {}
    for i, line in enumerate(open(path)):

        for match in re.finditer(pattern, line):
            indices_section[match.group()[4:]] = i+1

    pattern = re.compile("self.[A-Z_]* =")
    indices_args = {}
    for i, line in enumerate(open(path)):
        for match in re.finditer(pattern, line):
            indices_args[i+1] = match.group().split(' =')[0].split('.')[1]

    sections = {}
    for section, idx in indices_section.items():
        sections[section] = []
        stop_criterion = True
        i = idx + 1
        while stop_criterion:
            if i in indices_args:
                sections[section].append(indices_args[i])
                i += 1
            else:
                stop_criterion = False
    return sections