import cv2
import os
import copy
import argparse
import numpy as np

from .utils import *
from .bbox import BBoxMode
from ..config import mtqdm as tqdm


'''
Prepare dataset of large images into same size images with 
corresponding labels. 
- image_tile_size: size of the prepared images
- overlap: overlap between tiles (between 0 and 1)
- raw_dataset_folder: path to raw dataset
- dataset_path: path to prepared dataset
- dataset_name:
'''
class DataPreparator():
    def __init__(self,image_tile_size:int =256, 
                 overlap: float = 0.1, # between 0 and 1: how much overlapping between frames
                 box_overlap: float = 0.5,
                 raw_data_folder: str = './',
                 dataset_path: str = './prepared_dataset',
                 dataset_name: str = ''):

        self.image_tile_size = image_tile_size
        self.overlap = overlap
        self.box_overlap = box_overlap
        self.raw_data_folder = raw_data_folder
        self.dataset_path  = dataset_path
        self.dataset_name = dataset_name

        self.images_paths = []
        self.labels_paths = []

    def retrieve_images_paths(self):
        paths_to_explore = []
        images_folder = os.path.join(self.raw_data_folder, 'images')
        labels_folder = os.path.join(self.raw_data_folder, 'labelTxt')
        

        '''
        Assuming data is distributed in one folder or an arbitrary number of sub-folder with 
        depth 1 from raw_data_folder/images/ .
        '''
        if list_dir_only(images_folder) != []:
            paths_to_explore = list(map(lambda dir: dir, os.listdir(images_folder)))
        else:
            paths_to_explore = [""]

        for path in paths_to_explore:
            self.images_paths = self.images_paths + \
                                list(map(lambda file: os.path.join(images_folder, path, file),
                                         filter_images(os.listdir(os.path.join(images_folder, path)))))

            # self.labels_paths = self.labels_paths + \
            #                     list(map(lambda file: os.path.join(labels_folder, path,
            #                                                        change_file_name_extension(file, 'txt')), 
            #                              os.listdir(os.path.join(labels_folder, path))))

        self.labels_paths = list(
            map(lambda file: change_file_name_extension(file, 'txt'), self.images_paths))
    
    def prepare_dataset(self):
        # assert self.dataset_name == 'DOTA', 'Only DOTA dataset need to be resized'

        self.labels_paths.sort()
        self.images_paths.sort()

        self.retrieve_images_paths()
        self.patch_path = os.path.join(self.dataset_path, 'images')
        self.patch_labels_path = os.path.join(self.dataset_path, 'labelTxt')
        

        if not os.path.isdir(self.patch_path):
            os.makedirs(self.patch_path)
        if not os.path.isdir(self.patch_labels_path):
            os.makedirs(self.patch_labels_path)

        for idx, img_path in enumerate(tqdm(self.images_paths, desc='Dataset preparation')):
            # print('Processing {}...'.format(img_path))
            image = cv2.imread(img_path)
            
            # if image has at least one of its dimensions smaller than the tile size, skip it.
            if image.shape[0] < self.image_tile_size or image.shape[1] < self.image_tile_size:
                continue
            labels = get_object_from_label_file(
                self.labels_paths[idx], mode= self.dataset_name)
            patches, patches_labels = self.tile(image, labels)
            self.save_patches(patches, patches_labels, img_path)
            # return image, patches, patches_labels
    
    '''
    Tile the images and its labels. 
    Return a 2-d list of patches and a 2-d list of list of labels.
    When a box is not entierely in a patch, it is discarded. Hence some large objects
    may be lost when tiling with a too small image_tile_size.
    '''

    def tile(self, img, labels):
        H, W, _ = img.shape

        overlap_pix = int(self.overlap * self.image_tile_size)
        delta = self.image_tile_size - overlap_pix

        n_x = W // delta
        r_x = W % delta
        n_y = H // delta
        r_y = H % delta

        if r_x < overlap_pix:
            n_x -= 1
            r_x += delta
        
        if r_y < overlap_pix:
            n_y -= 1
            r_y += delta


        def get_column_patches(x_1, x_2):
            patch_col = []
            patch_labels_col = []
            for j in range(n_y):
                patch_col.append(
                    img[j*delta:j*delta + self.image_tile_size, x_1:x_2, :])
                patch_labels_col.append(self.find_bbox_in_patch(
                    labels, x_1, x_2, j*delta, j*delta + self.image_tile_size))

            if r_y != 0:
                keep = max(0, H - self.image_tile_size)
                patch_col.append(img[keep:, x_1:x_2, :])
                patch_labels_col.append(self.find_bbox_in_patch(
                    labels, x_1, x_2, keep, H))

            return patch_col, patch_labels_col


        patches = []
        patches_labels = []
        for i in range(n_x):
            patch_col, patch_labels_col = get_column_patches(i * delta, i * delta + self.image_tile_size)
            patches.append(patch_col)
            patches_labels.append(patch_labels_col)
        if r_x != 0:
            keep = max(0, W - self.image_tile_size)
            patch_col, patch_labels_col=get_column_patches(
                keep, W)

            patches.append(patch_col)
            patches_labels.append(patch_labels_col)
        return patches, patches_labels
    
    def find_bbox_in_patch(self, labels, x_1, x_2, y_1, y_2, labels_mode=BBoxMode.XYWH):
        labels_in_patch = []

        for label in labels:
            bbox = label['bbox']

            box_cliped = copy.copy(bbox)
            box_cliped.clip(x_1, y_1, x_2, y_2)
            if box_cliped.area() >= self.box_overlap * bbox.area():
                box_cliped.change_coord_origin(x_1, y_1)
                labels_in_patch.append({'bbox': box_cliped})

        
        return labels_in_patch

    def save_patches(self, patches, patches_labels, img_path): 
        for i in range(len(patches)):
            for j in range(len(patches[0])):
                if len(patches_labels[i][j]) > 0:
                    patch_name = '{}.{}.{}.jpg'.format(
                        img_path.split('/')[-1][:-4],
                        i,
                        j
                    )
                    cv2.imwrite(os.path.join(self.patch_path, patch_name), patches[i][j])

                    self.write_patch_labels(patch_name, patches_labels[i][j])
    
    def write_patch_labels(self, patch_name, labels):
        patch_labels_name = '{}.txt'.format(patch_name[:-4])
        labels_path = os.path.join(self.patch_labels_path, patch_labels_name)
        file = open(labels_path, 'w')
        for label in labels:
            bbox = label['bbox']
            line = bbox.get_line_cxywh(img_size=self.image_tile_size)
            file.write(line)       
        file.close()             


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-path', default='../data_prep_test/train')
    parser.add_argument('--tile-size', default=256, type=int)
    parser.add_argument('--overlap', default=0.1, type=float)
    parser.add_argument('--output-path', default='../data_prep_test')
    parser.add_argument('--dataset-name', default='DOTA')
    args = parser.parse_args()

    preparator = DataPreparator(raw_data_folder=args.raw_data_path,
                                dataset_path=args.output_path,
                                dataset_name=args.dataset_name, 
                                image_tile_size=args.tile_size, 
                                overlap=args.overlap)
    
    preparator.prepare_dataset()

