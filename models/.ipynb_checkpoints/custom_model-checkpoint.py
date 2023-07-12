import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn as original_fasterrcnn
from .faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models._utils import IntermediateLayerGetter
import torch.autograd.profiler as profiler
from .prototypical_classifier import NonPrototypicalRPNHead, PrototypicalRPNHead, PrototypicalFastRCNNPredictor
from .prototypical_rpn import PrototypicalRPN
from .prototypical_roi_heads import PrototypicalRoIHeads
from .backbone import Backbone
from ..data.utils import filter_annotations, draw_img_boxes_proposals
from .poolers import MultiScaleRoIAlign

from ..config import *

class CustomModel(torch.nn.Module):
    '''
    Wrapper for Faster RCNN model from torchvision.
    Get rid of the pre-processing transform that causes memory issue
    (resize 800x800) and get freedom on image normalization.
    '''
    def __init__(self, config):
        super(CustomModel, self).__init__()

        # self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        # model_temp = original_fasterrcnn(pretrained=False)
        # self.model.backbone = model_temp.backbone
        

        if config.PRETRAINED_BACKBONE == 'imagenet':
            self.model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True, 
                                        trainable_backbone_layers=config.BACKBONE_TRAINABLE_LAYERS)        
        else:
            self.model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

        # use default torchvision resnet50 for init to avoid FrozenBatchNorm
        # if config.SSL_PRETRAIN:
        #     resnet = torchvision.models.resnet50()
        #     return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        #     self.model.backbone.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        # self.model.backbone = Backbone()
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False
        self.config = config
        self.n_classes = self.config.N_CLASSES # includes background as a class here

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.n_classes)
        


    def forward(self, image_tensor, boxes_list=None, labels_list=None, keep_list=None):
        image_tensor = image_tensor.to(device, non_blocking=True)
        if self.training:
            assert boxes_list is not None and labels_list is not None, 'During training you must pass targets to network as well.'

            # when training with episodic training, some classes must be discarded from labels and boxes
            # sets as some classes are ignored. 

            if keep_list is not None:
                # filter boxes and labels with task selected classes
                boxes_list_task, labels_list_task = filter_annotations(boxes_list, labels_list, keep_list, mode='task_labels')
                boxes_list_all, labels_list_all = filter_annotations(boxes_list, labels_list, keep_list, mode='all_labels')

                boxes_scale_tensor = torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2).to(device)
                targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * boxes_scale_tensor,
                    'labels': labels_tensor.to(device, non_blocking=True),
                    'boxes_all': boxes_tensor_all.to(device, non_blocking=True) * boxes_scale_tensor,
                    'labels_all': labels_tensor_all.to(device, non_blocking=True)} 
                    for boxes_tensor, labels_tensor, boxes_tensor_all, labels_tensor_all in zip(boxes_list_task, 
                                                                                                labels_list_task, 
                                                                                                boxes_list_all, 
                                                                                                labels_list_all)]
            
            else:
                targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * boxes_scale_tensor,
                    'labels': labels_tensor.to(device, non_blocking=True),
                    'boxes_all': None,
                    'labels_all': None} 
                    for boxes_tensor, labels_tensor in zip(boxes_list, labels_list)]

            
            features = self.model.backbone(image_tensor)

            img_list = ImageList(
                image_tensor, [(t.shape[-2], t.shape[-1]) for t in image_tensor])

            
            proposals, losses_dict = self.model.rpn(img_list, features, targets)
            predictions = [] # when training only rpn 
            proposals = [p.detach() for p in proposals]
            # with torch.no_grad():

            predictions, reg_cla_losses = self.model.roi_heads(features,
                                                            proposals,
                                                            img_list.image_sizes,
                                                            targets)
            
            losses_dict.update(reg_cla_losses)
            return predictions, losses_dict
            
        else:
            with profiler.record_function("Backbone forward"):
                features = self.model.backbone(image_tensor)
            img_list = ImageList(
                image_tensor, [(t.shape[-2], t.shape[-1]) for t in image_tensor])
                
            with profiler.record_function("RPN forward"):
                proposals, losses_dict = self.model.rpn(
                    img_list, features)

            with profiler.record_function("ROI_HEAD forward"):
                predictions, reg_cla_losses = self.model.roi_heads(features,
                                                               proposals,
                                                               img_list.image_sizes)
            # print(predictions)
            return predictions

class CustomModelPrototypical(CustomModel):
    def __init__(self, config, prototypical_head=True):
        super(CustomModelPrototypical, self).__init__(config)

        self.k_shots = self.config.K_SHOTS
        self.n_ways_train = self.config.N_WAYS_TRAIN
        self.config = config
        
        in_features = self.model.backbone.out_channels

        self.replace_rpn()


        nb_anchors_per_location = self.model.rpn.anchor_generator.num_anchors_per_location()[0]

        if not self.config.USE_REGULAR_RPN:
            rpn_head = PrototypicalRPNHead(
                in_features, nb_anchors_per_location, self.config,
                anchor_generator=self.model.rpn.anchor_generator).to(device)
        else: 
            rpn_head = NonPrototypicalRPNHead(
                in_features, nb_anchors_per_location, self.config,
                anchor_generator=self.model.rpn.anchor_generator).to(device)
        
        self.model.rpn.head = rpn_head
        for m in rpn_head.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)

        if prototypical_head: 
            self.replace_roi_heads(in_features)
            self.model.roi_heads.box_predictor = PrototypicalFastRCNNPredictor(
                128, nb_anchors_per_location, self.config, self.model.rpn.head, self.model.roi_heads.box_head)
        
        self.model.rpn.initialize_embedding_loss()
        self.model.roi_heads.initialize_embedding_loss()
    
        if self.config.SSL_PRETRAIN:
            self.embedding_d_sim = 2048
            feature_width = self.config.SSL_IMG_SIZE // 16

            self.model.projection_mlp = nn.Sequential(nn.Linear(256 * feature_width ** 2, self.embedding_d_sim),
                                                nn.BatchNorm1d(self.embedding_d_sim),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(self.embedding_d_sim, self.embedding_d_sim),
                                                nn.BatchNorm1d(self.embedding_d_sim),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(self.embedding_d_sim, self.embedding_d_sim),
                                                nn.BatchNorm1d(self.embedding_d_sim)).to(device)
            
            self.model.prediction_mlp = nn.Sequential(nn.Linear(self.embedding_d_sim, 512),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(512, self.embedding_d_sim)).to(device)
        
        self.add_spectral_norm()
    

    def replace_rpn(self):
        # Default parameter from FasterRCNN class       
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn_head = self.model.rpn.head
        rpn_anchor_generator = self.model.rpn.anchor_generator
        self.model.rpn = PrototypicalRPN(self.config, rpn_anchor_generator, rpn_head,
                                         rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                         rpn_batch_size_per_image, rpn_positive_fraction,
                                         rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
    
    def replace_roi_heads(self, out_channels):
        # Change here
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 128
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        box_predictor = None
        box_score_thresh = 0.2 # 0.05 regular value
        box_nms_thresh = 0.5 # 0.5 regular value
        box_detections_per_img = 100
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.5
        bbox_reg_weights = None

        self.model.roi_heads = PrototypicalRoIHeads(
            self.config,
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)


    def forward(self, image_tensor, boxes_list=None, labels_list=None, keep_list=None):
        image_tensor = image_tensor.to(device, non_blocking=True)
        if self.training:
            assert boxes_list is not None and labels_list is not None, 'During training you must pass targets to network as well.'

            # when training with episodic training, some classes must be discarded from labels and boxes
            # sets as some classes are ignored. 

            if keep_list is not None:
                # filter boxes and labels with task selected classes
                boxes_list_task, labels_list_task = filter_annotations(boxes_list, labels_list, keep_list, mode='task_labels')
                boxes_list_all, labels_list_all = filter_annotations(boxes_list, labels_list, keep_list, mode='all_labels')

                boxes_scale_tensor = torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2).to(device)
                targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * boxes_scale_tensor,
                    'labels': labels_tensor.to(device, non_blocking=True),
                    'boxes_all': boxes_tensor_all.to(device, non_blocking=True) * boxes_scale_tensor,
                    'labels_all': labels_tensor_all.to(device, non_blocking=True)} 
                    for boxes_tensor, labels_tensor, boxes_tensor_all, labels_tensor_all in zip(boxes_list_task, 
                                                                                                labels_list_task, 
                                                                                                boxes_list_all, 
                                                                                                labels_list_all)]
            
            else:
                targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * boxes_scale_tensor,
                    'labels': labels_tensor.to(device, non_blocking=True),
                    'boxes_all': None,
                    'labels_all': None} 
                    for boxes_tensor, labels_tensor in zip(boxes_list, labels_list)]

            features = self.model.backbone(image_tensor)

            img_list = ImageList(
                image_tensor, [(t.shape[-2], t.shape[-1]) for t in image_tensor])
            proposals, losses_dict = self.model.rpn(img_list, features, targets)
            boxes = [t['boxes'] for t in targets]

            # draw_img_boxes_proposals(image_tensor, boxes, proposals, self.config.DATASET_META)
            predictions = [] # when training only rpn 
            proposals = [p.detach() for p in proposals]
            # with torch.no_grad():
            predictions, reg_cla_losses = self.model.roi_heads(features,
                                                            proposals,
                                                            img_list.image_sizes,
                                                            targets, image_tensor=image_tensor)
            if reg_cla_losses is not None:
                losses_dict.update(reg_cla_losses)
            return predictions, losses_dict
            
        else:
            features = self.model.backbone(image_tensor)
            img_list = ImageList(
                image_tensor, [(t.shape[-2], t.shape[-1]) for t in image_tensor])

            proposals, losses_dict, scores = self.model.rpn(
                img_list, features, output_scores=True)

            predictions, reg_cla_losses = self.model.roi_heads(features,
                                                               proposals,
                                                               img_list.image_sizes,
                                                               scores_proposals=scores)
            # print(predictions)
            return predictions


    def build_prototypes(self, support_set, backbone):
        self.model.rpn.head.build_prototypes(support_set, backbone)
        self.model.roi_heads.box_predictor.build_prototypes()
        # self.model.roi_heads.box_predictor.build_prototypes(support_set, backbone, self.model.roi_heads.box_roi_pool)
    
    def add_spectral_norm(self):
        if self.config.SPECTRAL_NORM:
            for m in self.model.modules():
                if hasattr(m, 'weight') and ('Conv'in m.__class__.__name__ or 'Linear' in m.__class__.__name__):
                    m = nn.utils.spectral_norm(m) 
    
    def remove_spectral_norm(self):
        if self.config.SPECTRAL_NORM:
            for m in self.model.modules():
                if hasattr(m, 'weight') and ('Conv'in m.__class__.__name__ or 'Linear' in m.__class__.__name__):
                    m = nn.utils.spectral_norm.remove_spectral_norm(m) 