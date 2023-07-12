#!/usr/bin/env python -W ignore::DeprecationWarning

import sys
import math
import torch
import numpy as np
import torchvision
import optuna
import random
# from apex import amp

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList

from collections import deque

from ..models.custom_model import CustomModel

from ..config import *
from ..data.dataset import ObjectDetectionDataset
from ..eval.eval import EvaluatorCOCO, EvaluatorVOC
from ..data.bbox import BBoxMode
from ..models.load_model import *

'''
Trainer object to perform training of object detection models.
'''
class Trainer():
    def __init__(self,
                 logger=None,
                 trial=None,
                 config=None,
                 save_model_checkpoint_path='model_checkpoint.pt',
                 save_model_path=None,
                 dataset=None, 
                 initialize_model=True):

        self.save_model_checkpoint_path = save_model_checkpoint_path
        self.save_model_path = config.SAVE_PATH

        self.logger = logger
        self.trial = trial
        self.config = config

        self.nb_iter = 0
        self.epoch = 0

        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
        # torch.backends.cudnn.deterministic = True
        if not self.config.DEBUG:
            self.logger.bot.new_training(self.config, dataset=dataset)

        if initialize_model:
            if dataset is None:
                self.dataset = ObjectDetectionDataset(
                    self.config,
                    train_val_ratio=.997,
                    bbox_modes=[BBoxMode.XYXY, BBoxMode.REL])
            else:
                self.dataset = dataset
            self.train_set, self.validation_set = self.dataset.get_dataloaders(batch_size=16,
                                                                                num_workers=8)
            # print(next(iter(self.train_set))[1][0][0])

            self.total_iter = len(self.train_set)

            self.model = CustomModel(self.config)
            # print('Build optimizer in train.py')
            if self.config.OPTIMIZER == 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.LR,
                                                momentum=0.9, weight_decay=0.0001)
            elif self.config.OPTIMIZER == 'Adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LR,
                                                weight_decay=0.00001)

            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.config.SCHEDULING_STEP,
                                                        gamma=self.config.GAMMA)
            
            self.evaluatorCOCO = EvaluatorCOCO(
                self.validation_set, self.model, self.config.N_CLASSES - 1)
            self.evaluatorVOC = EvaluatorVOC(
                self.validation_set, self.model, self.config.N_CLASSES - 1) # remove bg class
            self.model.to(device)

            # Enable mixed precision with amp
            # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.config.OPT_AMP)

    '''
    Training loop
    '''
    def train(self):
        self.running_losses = deque(maxlen=100)
        for epoch in range(self.config.N_EPOCH):
            self.model.train()
            for batch_idx, (image_tensor, boxes_list, labels_list, keep_list, indices) \
                    in enumerate(mtqdm(self.train_set, desc='Epoch {}/{}'.format(epoch+1, self.config.N_EPOCH))):

                self.epoch = epoch
                
                # in case data augmentation (affine transform) removes all boxes from image
                if 0 in [label.shape[0] for label in labels_list]:
                    continue

                loss = self.train_step(image_tensor, boxes_list, labels_list, keep_list)
                self.running_losses.append(loss.item())

                if (batch_idx + 1) % self.config.SAVE_INTERVAL == 0:
                    self.save_model()

                if (batch_idx + 1) % self.config.TEST_INTERVAL == 0:
                    coco_ap, voc_ap = self.validation_step()
            coco_ap, voc_ap = self.validation_step()
        if not self.config.DEBUG:    
            self.logger.writer.add_hparams(vars(self.config), {'COCO AP': coco_ap})
            self.logger.bot.training_end(voc_ap)
        return coco_ap
            
    def train_step(self, image_tensor, boxes_list, labels_list, keep_list):
        '''
        Trainin step: load images on GPU, perform a forward pass and backpropagate.
        Return loss dict.
        '''

        predictions, losses_dict = self.model(image_tensor, boxes_list, labels_list, keep_list)

        loss = sum(loss for loss in losses_dict.values())
        losses_dict['Total Loss'] = loss

        self.log_losses(losses_dict)

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            raise optuna.exceptions.TrialPruned()
        
        if loss != 0:
            self.optimizer.zero_grad()
            loss.backward()

            # for idx, (name, param) in enumerate(self.model.model.named_parameters()):
            #     if param.grad is not None:
            #         # print(name, param.grad.norm())
            #         if self.config.OUTPUT_GRAD:
            #             allowed_names = ['body.conv1', 'layer1.0.conv1', 'layer4.2.conv1', 'fpn.layer_blocks.3', 'rpn', 'roi_heads']
            #             if any([(s in name and 'weight' in name) for s in allowed_names]):
            #                 self.logger.writer.add_histogram(name, param, self.nb_iter)
            #                 self.logger.writer.add_scalar('Gradient/{}'.format(name), param.grad.norm(), self.nb_iter)
            #     else:
            #         # print(name, 'is None')
            #         pass
                

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1) 
            self.optimizer.step()
            self.lr_scheduler.step()

        self.nb_iter += 1

        return loss
        

    def validation_step(self):
        self.model.eval()
        average_ap_coco, average_ap_voc = 0, 0
        with torch.no_grad():
            self.evaluatorCOCO = EvaluatorCOCO(
                self.validation_set, self.model, self.config.N_CLASSES)
            self.evaluatorVOC = EvaluatorVOC(
                self.validation_set, self.model, self.config.N_CLASSES)
            
            # average_ap_coco = self.evaluatorCOCO.eval()
            average_ap_voc = self.evaluatorVOC.eval()
        
        self.trial.report(average_ap_coco, self.nb_iter)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if not self.config.DEBUG:
            self.logger.writer.add_scalar(
                'Eval/CocoAP', average_ap_coco, self.nb_iter)
            self.logger.writer.add_scalar(
                'Eval/VOC AP', average_ap_voc, self.nb_iter)

        # Send notification through telegram chat
        self.logger.bot.log(self.epoch, self.config.N_EPOCH, self.nb_iter %
                                 self.total_iter, self.total_iter, np.mean(self.running_losses), average_ap_voc)
        self.logger.bot.check_updates()

        self.model.train()
        return average_ap_coco, average_ap_voc
    
    def training_checkpoint(self, epoch):
        print('Checkpoint saving')
        state = {'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 
                'scheduler': self.lr_scheduler.state_dict()}

        torch.save(state, self.save_model_checkpoint_path)
    
    def resume_from_checkpoint(self):
        print('Resuming training')
        state = torch.load(self.save_model_checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['scheduler'])
        return state['epoch']
    
    def save_model(self):
        model_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(self.nb_iter))
        save_model(self.model, self.config, model_path)

    
    def log_losses(self, losses_dict):
        if not self.config.DEBUG:
            for key, value in losses_dict.items():
                self.logger.writer.add_scalar('Losses/' + key, value, self.nb_iter)
        
        
if __name__ == "__main__":
    trainer = Trainer(config=cfg)
    trainer.train()    
