import torch.nn.functional as F
import numpy as np
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning import distances as dist_metrics
from torchvision.models.detection import _utils as det_utils
from .rpn import *
from .prototypes_utils import pad_list
from .losses import *
from ..config import device
from .utils import balance_classes

import time
import cv2
import random

class PrototypicalRPN(RegionProposalNetwork):
    def __init__(self, config, *args):
        super().__init__(*args)

        self.config = config
        self.negative_matcher = det_utils.Matcher(0.5, 0.1,
            allow_low_quality_matches=True)

        self.focal_loss_fn = FocalLoss(gamma=0.0, alpha=0.5)
        self.embedding_loss = None
    
    def assign_targets_to_anchors(self, anchors, targets, which_boxes='boxes'):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        matched_idxs_list = []

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image[which_boxes]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(
                    anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros(
                    (anchors_per_image.shape[0],), dtype=torch.float32, device=device)
                matched_idxs = torch.zeros(
                    (anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes, anchors_per_image)
                true_match = match_quality_matrix.max(dim=0)[0]

                if which_boxes == 'boxes':
                    matched_idxs = self.proposal_matcher(match_quality_matrix)
                elif which_boxes == 'boxes_all':
                    matched_idxs = self.negative_matcher(match_quality_matrix)

                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(
                    min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            matched_idxs_list.append(matched_idxs)
        return labels, matched_gt_boxes, matched_idxs_list#, true_match
    
    def compute_loss(self, objectness,
                         pred_bbox_deltas,
                         labels,
                         regression_targets,
                         classification_targets,
                         embeddings, labels_neg,
                         probabilities=None,
                         distances=None,
                         sigmas=None):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Arguments:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        # initialize embedding loss
        if self.embedding_loss is None:
            self.initialize_embedding_loss()

        ratio = (labels[0] > 0).sum() / (labels[0].numel())
        # print((labels[0] > 0).sum(), ratio)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        hard_neg_inds = torch.where(labels_neg[0] > 0)[0]

        print('\nNumber of positive boxes in RPN: {}'.format(sampled_pos_inds.numel()))

        gt_labels = torch.cat(classification_targets)

        sampled_pos_inds = balance_classes(sampled_pos_inds, gt_labels, max_item_per_class=self.config.RPN_EX_PER_CLASS)

        # keep 1:3 ratio between pos and neg examples
        if self.config.RPN_POS_NEG_RATIO != 'all':
            nb_keep_neg = min(sampled_pos_inds.shape[0] * self.config.RPN_POS_NEG_RATIO - hard_neg_inds.shape[0], 
                            sampled_neg_inds.shape[0])
            sampled_neg_inds = sampled_neg_inds[:nb_keep_neg]
        
        if self.config.RPN_HEM:
            sampled_inds = torch.cat([sampled_pos_inds, hard_neg_inds, sampled_neg_inds], dim=0)
        else:
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)


        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0).clamp(0.0, 1.0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction='sum',
        ) / (sampled_inds.numel())
        
        objectness_loss = 0
        
        if self.config.USE_REGULAR_RPN:
            objectness_loss += F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds],
                reduction='mean')
        else:
            objectness_loss += self.focal_loss_fn(objectness[sampled_inds],
                                             labels[sampled_inds])


        embedding_loss = 0
        if 'triplet' in self.config.RPN_EMBEDDING_LOSS:
            embedding_loss += self.embedding_loss(embeddings, targets=gt_labels, indices=sampled_pos_inds, modes= ['triplet'])

        if 'nll' in self.config.RPN_EMBEDDING_LOSS:
            embedding_loss += self.embedding_loss(modes=['nll'])
        
        if 'cos' in self.config.RPN_EMBEDDING_LOSS:
            embedding_loss += self.embedding_loss(modes=['cos'])


        return objectness_loss, box_loss * 10, embedding_loss

    def output_crop(self,idx):
        N = int(self.proposals_copy.numel() / 4 / len(self.images_copy.image_sizes))  # number of proposal per image
        box = self.proposals_copy[int(idx.item() / N), int(idx.item()) % N]
        x1, y1, x2, y2 = box.to(int)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, 384)
        y2 = min(y2, 384)
        cv2.imwrite('test.jpg', self.images_copy.tensors[0, :, y1:y2, x1:x2].permute(1, 2, 0).cpu().numpy()*255)

    def forward(self,
                images,       # type: ImageList
                features,     # type: Dict[str, Tensor]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                output_embeddings=False,
                output_scores=False):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        if output_embeddings:
            return self.head.forward_embeddings(images, features)


        objectness, pred_bbox_deltas, anchors, probabilities, embeddings = self.head(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [
            o[0].shape for o in pred_bbox_deltas]
        num_anchors_per_level = [s[0] * s[1] * s[2] // 4
                                 for s in num_anchors_per_level_shape_tensors]

        placeholder = [torch.zeros(bb.shape[0], bb.shape[1]//4, *bb.shape[-2:]) for bb in pred_bbox_deltas]
        _, pred_bbox_deltas = \
            concat_box_prediction_layers(placeholder, pred_bbox_deltas)

        # As in Faster R-CNN no backprop through objectness
        objectness = objectness.flatten()
        probabilities = probabilities.flatten(end_dim=1)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        self.images_copy = images
        self.proposals_copy = proposals

        boxes, scores = self.filter_proposals(
            proposals, objectness, probabilities, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            print('Nb of gt in image: {}'.format(targets[0]['boxes'].shape[0]))
            labels, matched_gt_boxes, matched_idxs = self.assign_targets_to_anchors(
                anchors, targets)
            # get boxes labels for negative hard examples: i.e. boxes that are in training classes set
            # but not in task classes set.     
            labels_neg, matched_gt_boxes_neg, matched_idxs_neg = self.assign_targets_to_anchors(
                anchors, targets, which_boxes='boxes_all')

            regression_targets = self.box_coder.encode(
                matched_gt_boxes, anchors)

            classification_targets = [torch.where(
                idx >= 0, t['labels'][idx.clamp(0)], torch.ones_like(idx) * -1) for t, idx in zip(targets, matched_idxs)]
                

            loss_objectness, loss_rpn_box_reg, loss_rpn_embedding = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets, 
                classification_targets, embeddings, labels_neg, 
                probabilities=probabilities, 
                distances=self.head.distances_stacked, 
                sigmas=self.head.sigmas)

            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_rpn_embeddings": loss_rpn_embedding
            }
        if output_scores:
            return boxes, losses, scores
        return boxes, losses

    def filter_proposals(self, proposals, objectness, probabilities, image_shapes, num_anchors_per_level, filter_per_class=False):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        probabilities = probabilities.detach()
        C = probabilities.shape[-1]
        probabilities = probabilities.reshape(num_images, -1, C)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        probabilities = probabilities[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        
        for boxes, scores, probas, lvl, img_shape in zip(proposals, objectness, probabilities, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, probas, lvl = boxes[keep], scores[keep], probas[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            if filter_per_class:
                keep = self.keep_boxes_per_classe(probas[keep], self.post_nms_top_n())
            else:
                keep = keep[:self.post_nms_top_n()]
            boxes, scores, probas, lvl = boxes[keep], scores[keep], probas[keep], lvl[keep]
            # boxes, scores, probas, lvl = self.prototypical_box_filtering(
            #     boxes, scores, probas, lvl, img_shape[0])
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def keep_boxes_per_classe(self, probas, top_n):
        C = probas.shape[-1]
        top_n_class = top_n // C
        keep_idx = []
        for c in range(C):
            keep_idx.append(torch.topk(probas[:,c], k=top_n_class)[1])
        return torch.cat(keep_idx)
    
    def prototypical_box_filtering(self, boxes, scores, probas, lvl, image_size):
        r = 3
        classes = self.head.get_classes_episode()
        prototypes_boxes = self.head.prototypes_boxes
        
        if boxes.shape[0] != 0:
            labels = probas.max(dim=-1)[1]
            labels = torch.Tensor(classes)[labels]
            max_proto_class = max([len(v)
                                    for v in prototypes_boxes.values()])

            proto_hw = {k: [(box[2:] - box[:2]) for box in v]
                        for k, v in prototypes_boxes.items()}
            proto_of_labels = [proto_hw[label.item()] for label in labels]

            
            hw_star_min = torch.ones(labels.shape[0], max_proto_class, 2).to(device) * np.inf
            hw_star_max = torch.ones(labels.shape[0], max_proto_class, 2).to(device) * -np.inf

            for idx_proto, proto in enumerate(proto_of_labels):
                hw_star_min[idx_proto, :len(proto), :] = 1 / r * torch.stack(proto)
                hw_star_max[idx_proto, :len(proto), :] = r * torch.stack(proto)

            hw_boxes = (boxes[:, 2:] - boxes[:, :2]).view(-1,
                                                            1, 2).expand_as(hw_star_max) / image_size

            condition = (hw_boxes > hw_star_min) * (hw_boxes < hw_star_max)
            keep = (condition.sum(dim=-1) == 2).sum(dim=-1) >= 1
            # print(boxes_list[idx].shape)
            boxes = boxes[keep]
            # print(boxes_list[idx].shape)
            probas = probas[keep]
            scores = scores[keep]
            labels = labels[keep]
            lvl = lvl[keep]

        return boxes, scores, probas, lvl
    
    def initialize_embedding_loss(self):
        self.embedding_loss = EmbeddingLoss(self.head)
