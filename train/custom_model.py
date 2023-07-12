import torch
import torchvision
from ..models.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.models.detection.image_list import ImageList

from ..config import *

'''
Wrapper for Faster RCNN model from torchvision.
Get rid of the pre-processing transform that causes memory issue
(resize 800x800) and get freedom on image normalization.
'''
class CustomModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(CustomModel, self).__init__()

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, n_classes)

    def forward(self, image_tensor, boxes_list=None, labels_list=None):
        image_tensor = image_tensor.to(device, non_blocking=True)
        if self.training:
            assert boxes_list is not None and labels_list is not None, 'During training you must pass targets to network as well.'

            boxes_scale_tensor = torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2).to(device)
            targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * boxes_scale_tensor,
                'labels': labels_tensor.to(device, non_blocking=True)} 
                for boxes_tensor, labels_tensor in zip(boxes_list, labels_list)]

            features = self.model.backbone(image_tensor)
            img_list = ImageList(
                image_tensor, [(t.shape[-2], t.shape[-1]) for t in image_tensor])

            proposals, losses_dict = self.model.rpn(img_list, features, targets)
            predictions, reg_cla_losses = self.model.roi_heads(features,
                                                            proposals,
                                                            img_list.image_sizes,
                                                            targets)
            
            losses_dict.update(reg_cla_losses)
            return predictions, losses_dict
            
        else:
            features = self.model.backbone(image_tensor)
            img_list = ImageList(
                image_tensor, [(t.shape[-2], t.shape[-1]) for t in image_tensor])

            proposals, losses_dict = self.model.rpn(
                img_list, features)

            predictions, reg_cla_losses = self.model.roi_heads(features,
                                                               proposals,
                                                               img_list.image_sizes)
            return predictions
