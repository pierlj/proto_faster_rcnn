import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_metric_learning import losses, distances
from .roi_heads import *
from .losses import *
from ..config import device
from .prototypes_utils import *
from .utils import balance_classes
from .post_process_detection import PostProcessDetection

from torchvision.models.detection import _utils as det_utils
from sklearn.cluster import KMeans


class PrototypicalRoIHeads(RoIHeads):
    def __init__(self, config, *args):
        super().__init__(*args)

        self.config = config
        self.negative_matcher = det_utils.Matcher(0.5, 0.1,
            allow_low_quality_matches=False)
        
        self.focal_loss_fn = FocalLoss(gamma=0.0, alpha=[1.0] + [1.0 for _ in range(config.N_WAYS_TRAIN)])
        self.embedding_loss = None

        self.post_processor = PostProcessDetection(self.box_coder)
        self.centers = None
    
    def compute_loss(self, class_logits,
                         box_regression,
                         labels,
                         regression_targets,
                         embeddings,
                         hard_negatives,
                         reconstructed_features, 
                         pooled_features):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Computes the loss for Faster R-CNN.

        Arguments:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        # initialize embedding loss
        if self.embedding_loss is None:
            self.initialize_embedding_loss()

        labels = torch.cat(labels, dim=0)
        hard_negatives = torch.cat(hard_negatives, dim=0)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        nb_pos_samples = sampled_pos_inds_subset.shape[0]
        sampled_neg_inds_subset = torch.where(labels == 0)[0]
        sampled_neg_inds_subset = sampled_neg_inds_subset[torch.randperm(
            sampled_neg_inds_subset.shape[0])]

        sampled_pos_inds_subset = balance_classes(sampled_pos_inds_subset, labels, max_item_per_class=self.config.ROI_EX_PER_CLASS)
        if self.config.ROI_POS_NEG_RATIO != 'all':
            sampled_neg_inds_subset = sampled_neg_inds_subset[ :self.config.ROI_POS_NEG_RATIO * nb_pos_samples]

        print('Number of positive boxes in RoI: {}'.format(sampled_pos_inds_subset.numel()))
        if (sampled_pos_inds_subset.numel() <= 1):
            print("h")

        hard_negatives_inds = torch.where(hard_negatives > 0)[0]

        if self.config.ROI_HEM:
            balanced_sampled = torch.cat([sampled_pos_inds_subset, 
                                    hard_negatives_inds, 
                                    sampled_neg_inds_subset])
        else:
            balanced_sampled = torch.cat([sampled_pos_inds_subset,
                                    sampled_neg_inds_subset])   

        #######################
        # Unseen cluster loss #
        #######################
        all_neg_indices = torch.where(labels == 0)[0]
        neg_embeddings = embeddings[all_neg_indices]
        pos_embeddings = embeddings[sampled_pos_inds_subset]
        
        # Compute clusters
        kmeans_predictor = KMeans(n_clusters=100, max_iter=100, 
                                init=self.centers if self.centers is not None else 'random')
        kmeans_predictor.fit(X=neg_embeddings.detach().cpu())
        cluster_labels, self.centers = kmeans_predictor.labels_, kmeans_predictor.cluster_centers_
        dist = distances.LpDistance()
        dist_to_unseen_proto = dist(embeddings, torch.from_numpy(self.centers).to(dtype=torch.float, device=device))
        probabilities_unseen = torch.exp(-dist_to_unseen_proto / 2 / self.config.ROI_HEAD_SIGMA)

        if self.config.KMEANS_GHOST_CLASSES:
            class_logits = class_logits / (class_logits.sum(dim=1, keepdim=True) + 
                                            10 * probabilities_unseen.sum(dim=1, keepdim=True))
        else:
            class_logits = class_logits / class_logits.sum(dim=1, keepdim=True)
                                        

        if self.config.KMEANS_NEG_CLUST:
            lambda_clust = 1
        else:
            lambda_clust = 0

        loss_cluster_fn = losses.NTXentLoss(temperature=0.5)

        cluster_loss = loss_cluster_fn(neg_embeddings[:200], 
                            torch.from_numpy(cluster_labels[:200]).to(device)) * lambda_clust

        #######################
        # Classification Loss #
        #######################
        # TO DO write conversion below in an utils file because it used several times

        labels = torch.where(
            labels.unsqueeze(-1) ==
            torch.cat([torch.zeros(1), torch.Tensor(self.box_predictor.rpn_classifier.get_classes_episode()) + 1]).to(device))[-1] \
            .view_as(labels)

        classification_loss = self.focal_loss_fn(class_logits[balanced_sampled], labels[balanced_sampled]) #* 0

        ###################
        # Regression Loss #
        ###################
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        regression_targets = torch.cat(regression_targets, dim=0)
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction='sum',
        )
        
        box_loss = box_loss / labels_pos.numel()

        ##################
        # Embedding Loss #
        ##################
        embedding_loss = 0

        if 'triplet' in self.config.ROI_HEAD_EMBEDDING_LOSS:
            embedding_loss += self.embedding_loss(embeddings, labels=labels, modes=['triplet'])

        elif 'nll' in self.config.ROI_HEAD_EMBEDDING_LOSS:
            embedding_loss += self.embedding_loss(modes=['nll'])
        
        elif 'cos' in self.config.ROI_HEAD_EMBEDDING_LOSS:
            embedding_loss += self.embedding_loss(modes=['cos'])

        #######################
        # Reconstruction Loss #
        #######################

        # L1 or L2? between reconstructed_features and pooled_features from rpn
        reconstruction_loss = F.l1_loss(reconstructed_features,
                                        pooled_features.flatten(start_dim=1))


        return classification_loss, box_loss / 4, embedding_loss, 1 * cluster_loss
    
    def compute_centers(self, embeddings, labels):
        labels_indices_list = {c: torch.where(labels == c)[0] for c in [0] + self.box_predictor.get_classes_episode()}
        centers = {
                    c : torch.mean(embeddings[labels_indices], dim=0) 
                                    for c, labels_indices in labels_indices_list.items() 
                                        if labels_indices.numel() > 0
                }

        return centers
        

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None,   # type: Optional[List[Dict[str, Tensor]]]
                output_embeddings=False, output_raw_predictions=False,
                scores_proposals=None,
                image_tensor=None):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                t['labels'] = t['labels'] + 1
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            if self.config.DEBUG and False:
                plot_best_proposals(proposals, targets, image_tensor)
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets)
            # print(proposals[0].shape)
            boxes_not_task = [t['boxes_all'] for t in targets]
            labels_not_task = [t['labels_all'] for t in targets]
            hard_negatives, labels_hard = self.assign_targets_to_proposals(proposals, boxes_not_task, labels_not_task)

        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features_pooled, levels_embeddings = self.box_roi_pool(features, proposals, image_shapes, return_levels=True)
        if box_features_pooled.shape[0] == 0:
            return None, None
        box_features = self.box_head(box_features_pooled)

        class_logits, box_regression, embeddings, reconstructed_features = self.box_predictor(box_features, levels_embeddings)


        if output_raw_predictions or output_embeddings:
            # class_logits = torch.nn.functional.softmax(class_logits, dim=-1)
            scores_proposals = torch.cat(scores_proposals)
            # pred_scores = torch.pow(class_logits, 2.0) * torch.pow(scores_proposals.unsqueeze(-1), 10.0)# * torch.pow(iou_pred, 2.0)
            pred_scores = torch.ones_like(class_logits) * torch.pow(scores_proposals.unsqueeze(-1), 1.0)

            pred_boxes = self.box_coder.decode(box_regression, proposals)
            pred_boxes = pred_boxes[:,1:].max(dim=1)[0].split(1000)
            pred_scores = pred_scores[:,1:].max(dim=-1)[0].split(1000)
            if output_embeddings:
                return embeddings, pred_boxes, pred_scores
            return pred_boxes, pred_scores
        
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None


            loss_classifier, loss_box_reg, loss_embeddings, loss_reconstruction = self.compute_loss(
                class_logits, box_regression, labels, regression_targets, embeddings, hard_negatives, reconstructed_features, box_features_pooled)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_embedding_classifier": loss_embeddings,
                "loss_reconstruction": loss_reconstruction
            }
        else:

            boxes, scores, labels = self.post_processor.postprocess(
                class_logits, box_regression, proposals, image_shapes, scores_proposals, config=self.config)

            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i] - 1,
                        "scores": scores[i],
                    }
                )
                
        return result, losses
    
    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes,     # type: List[Tuple[int, int]]
                               scores_proposals,
                               iou_pred):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0]
                           for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = class_logits
        if scores_proposals is not None:
            scores_proposals = torch.cat(scores_proposals)
            pred_scores = torch.pow(class_logits, 0.5) * torch.pow(scores_proposals.unsqueeze(-1), 0.5)


        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            # boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression
            keep = box_ops.batched_nms(boxes, scores, torch.ones_like(labels), 0.8)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep = batched_soft_nms(boxes, scores, labels, sigma=0.01, score_threshold=0.9)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def prototypical_postprocess(self, boxes_list, scores_list, labels_list, prototypes_boxes, image_size):
        r = 2
        classes = self.box_predictor.rpn_classifier.get_classes_episode()
        for idx, boxes in enumerate(boxes_list):
            if boxes.shape[0] != 0:
                labels = labels_list[idx] - 1
                labels = torch.Tensor(classes)[labels]
                max_proto_class = max([len(v) for v in prototypes_boxes.values()])

                proto_hw = {k: [(box[2:] - box[:2]) for box in v]
                            for k, v in prototypes_boxes.items()}
                proto_of_labels = [proto_hw[label.item()] for label in labels]

                hw_star_min = 1 / r * torch.cat([torch.stack(pad_list(v, torch.Tensor(
                    [np.inf, np.inf]).to(device), max_proto_class)) for v in proto_of_labels]).view(-1, max_proto_class, 2)

                hw_star_max = r * torch.cat([torch.stack(pad_list(v, -torch.Tensor([np.inf, np.inf]).to(device), max_proto_class))
                                for v in proto_of_labels]).view(-1, max_proto_class, 2)

                hw_boxes = (boxes[:, 2:] - boxes[:, :2]).view(-1,
                                                            1, 2).expand_as(hw_star_max) / image_size

                condition = (hw_boxes > hw_star_min) * (hw_boxes < hw_star_max)
                keep = (condition.sum(dim=-1) == 2).sum(dim=-1) >= 1 
                # print(boxes_list[idx].shape)
                boxes_list[idx] = boxes_list[idx][keep]
                # print(boxes_list[idx].shape)
                scores_list[idx] = scores_list[idx][keep]
                labels_list[idx] = labels[keep]

        return boxes_list, scores_list, labels_list
    
    def initialize_embedding_loss(self):
        self.embedding_loss = EmbeddingLoss(self.box_predictor)
