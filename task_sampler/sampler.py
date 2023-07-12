import numpy as np
import torch
import os
import json
import functools
import random

from ..data.dataset import ObjectDetectionDataset
from ..data.bbox import BBoxMode
from ..data.utils import change_file_name_extension
from ..data.dataset_categories import *

from ..config import mtqdm

class TaskSampler():
    def __init__(self, dataset, classes, config):
        '''
        Args: 
            - dataset: DatasetMeta object of the dataset used for training
            - classes: list of all classes
            - config: config object for the current training 
        '''

        self.classes = classes
        self.dataset_path = dataset.path
        self.dataset_name = dataset.name

        self.config = config
        self.config.DA_COLOR = 0.0
        self.config.DA_AFFINE = 0.0
        self.config.DA_HFLIP = 0.0
        self.config.DA_VFLIP = 0.0

        self.classes_table = {}

        self.same_classes = False

        try:
            self.load_classes_table()
        except:
            self.create_classes_table()
        
        self.split_train_test_classes(config.SPLIT_METHOD)
    
    def split_train_test_classes(self, split_method='random'):
        if split_method =='random':
            self.c_test = random.sample(self.classes, k=self.config.N_CLASSES_TEST)
            self.c_test.sort()
            self.c_train = [c for c in self.classes if c not in self.c_test]

        elif split_method == 'same':
            assert self.config.N_WAYS_TEST == self.config.N_WAYS_TRAIN, 'N_WAYS_TRAIN and N_WAYS_TEST should be equal when using "same" samling'
            self.c_test = random.sample(self.classes, k=self.config.N_CLASSES_TEST)
            self.c_test.sort()
            self.c_train = self.c_test

        elif split_method == 'deterministic':
            with open(os.path.join(self.dataset_path, 'classes_split.txt'), 'r') as f:
                lines = f.readlines()
                assert len(lines)== 2, 'Wrong classes split file format, should be: \ntrain_classes:1,2,3 \n test_classes:4,5,6'
                self.c_train = list(map(lambda x: int(x), lines[0][:-1].split(':')[-1].split(',')))
                self.c_test = list(map(lambda x: int(x), lines[1][:-1].split(':')[-1].split(',')))
                print(self.c_train, self.c_test)
                assert len(self.c_train) >= self.config.N_WAYS_TRAIN, 'N_WAYS_TRAIN too large for number of training classes defined in file'
                assert len(self.c_test) >= self.config.N_WAYS_TEST, 'N_WAYS_TEST too large for number of training classes defined in file'

                self.c_train.sort()
                self.c_test.sort()

    def sample_train_val_tasks(self, n_ways_train, n_ways_test, k_shots, n_query_train, n_query_test, verbose=False):
        '''
        Create two tasks for the episode (train and test) by sampling classes within the allowed range.

        Args:
        - n_ways_train: number of classes for training task
        - n_ways_test: number of classes for testing task
        - k_shots: number of examples for each class in the support set (both train and test)
        - n_query_train: number of image for each class in the train query set
        - n_query_test: number of image for each class in the train query set
        - verbose: outputs information about classes selected for the current tasks

        Returns:
        - train_task: Query set, Support set, task_classes
        - test_task: Query set, Support set, task_classes
        '''
        train_task = self.sample_task(
            n_ways_train, k_shots, n_query_train, self.c_train, verbose=verbose)
        test_task = self.sample_task(
            n_ways_test, k_shots, n_query_test, self.c_test, verbose=verbose)
        return train_task, test_task

    def sample_task(self, n_ways, k_shots, n_query, classes_from, verbose=False):
        '''
        Sample classes for a task and create dataset objects for support and query sets. 
        
        Args:
        - n_ways: number of classes for the task
        - k_shots: number of examples for the support set
        - n_query: number of examples for the query set
        - classes_from: classes list from which classes are chosen
        - verbose: display selected classes for current task
        '''
        assert n_ways <= len(classes_from), 'Not enough classes for this task, either n_ways is too big or classes_from is too small'

        self.c_episode = random.sample(classes_from, k=n_ways)

        self.c_episode.sort()
        if verbose:
            print('Sampling new task {} ways {} shots\nSelected classes: {}'.format(
                n_ways, k_shots, self.c_episode))

        support_paths = [random.sample(self.classes_table[c], k=k_shots) for c in self.c_episode]
        support_images = [
            path for class_path in support_paths for path in class_path]

        query_paths = [random.sample(
            self.classes_table[c], k=n_query) for c in self.c_episode]
        query_images = [
            path for class_path in query_paths for path in class_path]

        support_set = self.create_dataset_from_file_list(support_images, self.c_episode, k_shots_support=k_shots)
        query_set = self.create_dataset_from_file_list(query_images, self.c_episode, 
                                                all_classes_train=[c for c in self.classes if c not in self.c_test])

        return query_set, support_set, self.c_episode

    def create_dataset_from_file_list(self, file_list, c_episode, k_shots_support=0, all_classes_train=None):
        '''
        Create a dataset object from a list of annotations files

        Args:
        - file_list: list of annotations files for the current dataset
        - c_episode: classes selected for the current episode
        - k_shots_support: for support set, the allowed classes must be repeated as only one annotation
          (and therefore class) is allowed per image. 
        - all_classes_train: set of all classes from which each training episode draws its classes
        '''
        files_paths = list(map(lambda file: (os.path.join(
            self.dataset_path, 'images', change_file_name_extension(file, 'jpg')),
            os.path.join(self.dataset_path, 'labelTxt', file)), file_list
            ))
        
        # for support set, target allowed is defined per image, and therefore repeated k_shots times
        if k_shots_support > 0:
            allowed_classes = torch.Tensor(c_episode).unsqueeze(1).repeat((1, k_shots_support)).flatten()
        else:
            allowed_classes = torch.Tensor(c_episode)

        if all_classes_train is not None:
            all_classes_train = torch.Tensor(all_classes_train)

        dataset = ObjectDetectionDataset(
            self.config, train_val_ratio=1.0, 
            bbox_modes=[BBoxMode.XYXY, BBoxMode.REL], files_paths=files_paths, 
            target_allowed=allowed_classes, support_set=(k_shots_support > 0), 
            all_classes_train=all_classes_train)

        return dataset

    def create_classes_table(self):
        temp_annots = os.listdir(os.path.join(self.dataset_path, 'labelTxt'))
        annotation_paths = []
        for path in temp_annots:
            if os.path.isdir(os.path.join(self.dataset_path, 'labelTxt', path)):
                annotation_paths = annotation_paths + list(map(lambda file: os.path.join(path, file),
                    os.listdir(os.path.join(self.dataset_path, 'labelTxt', path))))
            else:
                annotation_paths.append(path)

        self.classes_table = {c:set([]) for c in self.classes}
        for file in mtqdm(annotation_paths):
            label_file = open(os.path.join(self.dataset_path, 'labelTxt', file))
            labels_lines = label_file.readlines()
            label_file.close()

            # Assuming label file is in format 'class x y w h'
            for line in labels_lines:
                # if line.split(' ')[0] == '':
                #     print('h')
                classe = int(line.split(' ')[0])
                self.classes_table[classe].add(file)
        
        output_table = {str(k): list(v) for k, v in self.classes_table.items()}
        with open(os.path.join(self.dataset_path, 'classes_table.json'), 'w') as f:
            json.dump(output_table, f)

        self.classes_table = {k: list(v) for k, v in self.classes_table.items()}
    
    def load_classes_table(self, file_name='classes_table.json'):
        with open(os.path.join(self.dataset_path, file_name)) as json_file:
            classes_table = json.load(json_file)
        
        self.classes_table = {int(k): v for k, v in classes_table.items()}

        ''' Repeat files' paths in table for classes that do not have enough examplars 
            so that it is convenient to sample from it
        '''
        for k,v in self.classes_table.items():
            if len(v) < self.config.N_QUERY_TEST or len(v) < self.config.N_QUERY_TRAIN:
                repeat = max(self.config.N_QUERY_TEST, self.config.N_QUERY_TRAIN) // len(v) + 1
                self.classes_table[k] = v * repeat
    
    def display_classes(self):
        train_set_classes = ', '.join([str(c) + " " + str(c) for c in self.c_train])
        val_set_classes = ', '.join([str(c) + " " + str(c)
                             for c in self.c_test])
        print('''Selected categories:
                Train support set: {}
                Validation support set: {}'''.format(train_set_classes, val_set_classes))
