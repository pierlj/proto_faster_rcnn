import sys
import math
import torch
import numpy as np
import torchvision
import optuna
import random
from apex import amp

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList

from collections import deque

from ..models.custom_model import CustomModelPrototypical

from ..config import *
from ..eval.eval import EvaluatorCOCO, EvaluatorVOC
from ..data.bbox import BBoxMode
from ..data.dataset import *
from ..task_sampler.sampler import TaskSampler
from .train import Trainer
from ..data.utils import filter_annotations
from .simclr_episode import SimCLRTrainer
from ..models.load_model import load_model

class EpisodicTrainer(Trainer):
    def __init__(self, 
                n_ways_train=5, 
                n_ways_test=5,
                k_shots=1,  # nb of image of each class in support set
                n_query_train=1,
                n_query_test=20,
                **kwargs):
        # print(args)
        super(EpisodicTrainer, self).__init__(initialize_model=False, **kwargs)

        self.n_ways_train = n_ways_train
        self.n_ways_test = n_ways_test
        self.k_shots = k_shots
        self.dataset_path = self.config.DATASET_META.path
        self.dataset_name = self.config.DATASET_META.name
        self.n_query_train = n_query_train
        self.n_query_test = n_query_test

        classes = [i for i in range(self.config.N_CLASSES - 1)] # background is class 0 but not in classes_table

        self.task_sampler = TaskSampler(self.config.DATASET_META, 
                                        classes,
                                        self.config)
        
        # self.model = CustomModel(self.config.N_CLASSES)
        self.model = CustomModelPrototypical(self.config)
        if self.config.PRETRAINED_BACKBONE is not None and os.path.isfile(self.config.PRETRAINED_BACKBONE):
            model_ssl = load_model(self.config.PRETRAINED_BACKBONE, self.config, load_pretrain_ssl=True)
            self.model.model.backbone = model_ssl.model.backbone
            # trainable_layers = 5
            # layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
            # for name, param in self.model.model.backbone.named_parameters():
            #     if all([not name.startswith(layer) for layer in layers_to_train]):
            #         param.requires_grad_(False)

        print('Build optimizer in train_ep.py')
        if self.config.OPTIMIZER == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.LR,
                                    momentum=0.9, weight_decay=0.0005)
        elif self.config.OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LR)
        elif self.config.OPTIMIZER == 'AdamW':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LR,
                                    weight_decay=0.01)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                            milestones=self.config.SCHEDULING_STEP,
                                            gamma=self.config.GAMMA)
        
        self.model.to(device)
        self.simclr_trainer = SimCLRTrainer(self.config, self.model.model)
        print(self.model)

    def train(self):
        self.running_losses = deque(maxlen=100)
        self.nb_iter = 0

        if self.config.SSL:
            sim_clr_dataset = SimCLRDataset(
                    self.model.config,
                    train_val_ratio=1.0,
                    bbox_modes=[BBoxMode.XYXY, BBoxMode.REL])
            sim_clr_loader = sim_clr_dataset.get_dataloaders(batch_size=4, num_workers=16)

        if not self.config.DEBUG:
            self.logger.writer.add_hparams(self.config.to_dict(), 
                        {'VOC AP Train': 0, 
                        'VOC AP Validation': 0})

        if not self.config.SSL_PRETRAIN:
            for self.epoch in range(self.config.N_EPOCH):
                self.model.train()
                
                (Q_train, S_train, classes_train), (Q_val, S_val, classes_val) = self.task_sampler. \
                    sample_train_val_tasks(self.n_ways_train,
                                    self.n_ways_test,
                                    self.k_shots,
                                    self.n_query_train,
                                    self.n_query_test, verbose=True)
                
                
                query_loader = Q_train.get_dataloaders(
                    batch_size=self.config.BATCH_SIZE)
                
                self.total_iter = len(query_loader)

                # self.model.build_prototypes(
                #             S_train, self.model.model.backbone)

                for batch_idx, (image_tensor, boxes_list, labels_list, keep_list, indices) \
                        in enumerate(mtqdm(query_loader, desc='Episode {}/{}'.format(self.epoch+1, self.config.N_EPOCH))):

                    # with torch.no_grad():
                    self.model.build_prototypes(
                        S_train, self.model.model.backbone)
                            
                    # in case data augmentation (affine transform) removes all boxes from image
                    if 0 in [label.shape[0] for label in labels_list]:
                        continue
                    
                    loss = self.train_step(image_tensor, boxes_list, labels_list, keep_list)

                    if self.config.SSL:
                        loss_simclr = self.train_ssl(*next(iter(sim_clr_loader))[:-1])

                    self.running_losses.append(loss.item())

                    if (self.nb_iter + 1) % self.config.SAVE_INTERVAL == 0:
                        self.save_model()

                    if (self.nb_iter + 1) % self.config.TEST_INTERVAL == 0:
                        voc_ap_train = self.validation_step(Q_train, S_train, classes_train, validation=False)
                        voc_ap_val = self.validation_step(Q_val, S_val, classes_val)
                        with torch.no_grad():
                            self.model.build_prototypes(S_train, self.model.model.backbone)

            if not self.config.DEBUG:
                self.logger.writer.add_hparams(self.config.to_dict(), 
                            {'VOC AP Train': voc_ap_train, 
                            'VOC AP Validation': voc_ap_val})

        else:
            for self.epoch in range(self.config.N_EPOCH): 
                for batch_idx, (image_tensor, boxes_list, labels_list, _) \
                    in enumerate(mtqdm(sim_clr_loader, desc='Episode {}/{}'.format(self.epoch+1, self.config.N_EPOCH))):

                    loss_simclr = self.train_ssl(image_tensor, boxes_list, labels_list)
                    self.nb_iter += 1
                    if (self.nb_iter + 1) % self.config.SAVE_INTERVAL == 0:
                        self.save_model()
                    voc_ap_val = 0
        return voc_ap_val

    def validation_step(self, Q_val, S_val, classes_val, validation=True):        
        with torch.no_grad():
            keep_prototypes = { c: [p.clone() for p in protos]
                for c,protos in self.model.model.roi_heads.box_predictor.prototypes.items()}
            self.model.eval()
            self.model.build_prototypes(S_val, self.model.model.backbone)
            
            Q_loader = Q_val.get_dataloaders(batch_size=self.config.BATCH_SIZE)
            self.evaluatorVOC = EvaluatorVOC(
                Q_loader, self.model, len(classes_val), few_shot_classes=classes_val)  # remove bg class
            
            average_ap_voc = 0
        

            average_ap_voc = self.evaluatorVOC.eval()

        self.trial.report(average_ap_voc, self.nb_iter)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if not self.config.DEBUG:
            if validation:
                self.logger.writer.add_scalar(
                    'Eval/VOC AP Validation', average_ap_voc, self.nb_iter)
            else: 
                self.logger.writer.add_scalar(
                    'Eval/VOC AP Train', average_ap_voc, self.nb_iter)

        if (self.nb_iter + 1) % self.config.LOG_INTERVAL == 0:
            self.logger.bot.log(self.epoch, self.config.N_EPOCH, self.nb_iter %
                                self.total_iter, self.total_iter, np.mean(self.running_losses), average_ap_voc)
            self.logger.bot.check_updates()

        self.model.model.roi_heads.box_predictor.prototypes = keep_prototypes
        self.model.train()
        print('End eval', self.model.training)

        return average_ap_voc

    # To be changed maybe
    def load_model(self, model_path):
        self.model = torch.load(model_path)
    
    def train_ssl(self, image_tensor, boxes_list, labels_list):
        loss = self.simclr_trainer.train_step(image_tensor, boxes_list)
        losses_dict = {'Sim SSL': loss}

        self.log_losses(losses_dict)

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            raise optuna.exceptions.TrialPruned()
        
        self.optimizer.zero_grad()
        loss.backward()      

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1) 
        self.optimizer.step()

