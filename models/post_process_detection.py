import torch
import numpy as np
import torchvision.ops.boxes as box_ops

from sklearn.cluster import MeanShift, estimate_bandwidth

from ..config import Config

class PostProcessDetection():
    def __init__(self, box_coder):
        self.box_coder = box_coder
    
    def postprocess(self, class_logits,    # type: Tensor
                        box_regression,  # type: Tensor
                        proposals,       # type: List[Tensor]
                        image_shapes,     # type: List[Tuple[int, int]]
                        scores_proposals,
                        config=Config()):

        self.config = config
        self.nb_prototypes = class_logits.shape[-1] - 1 # class_logits is N x n_classes with background counted

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        pred_boxes_list, pred_scores_list = self.prepare_predictions(class_logits, 
                                                                                    box_regression, 
                                                                                    proposals, 
                                                                                    image_shapes, 
                                                                                    scores_proposals)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes, scores, labels = self.prepare_boxes(boxes, scores, image_shape, num_classes, device)

            if self.config.POST_PROCESSING_MODE == 'frcnn':
                boxes, scores, labels = self.frcnn_postprocess(boxes, scores, labels)
            elif self.config.POST_PROCESSING_MODE == 'cluster':
                boxes, scores, labels = self.cluster_postprocess(boxes, scores, labels)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def frcnn_postprocess(self, boxes, scores, labels):
        
        # remove low scoring boxes
        inds = torch.where(scores > self.config.BOX_SCORE_THRESH)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression
        keep = box_ops.batched_nms(boxes, scores, torch.ones_like(labels), self.config.BOX_NMS_THRESH)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        
        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.config.BOX_NMS_THRESH_CLS)
        # keep = batched_soft_nms(boxes, scores, labels, sigma=0.01, score_threshold=0.9)
        # keep only topk scoring predictions
        keep = keep[:self.config.DETECTION_PER_IMAGE]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]


        return boxes, scores, labels

    def cluster_postprocess(self, boxes, scores, labels):
        boxes = boxes.cpu().detach()
        scores = scores.cpu().detach()
        labels = labels.cpu().detach()

        if self.config.CLUST_BW is None:
            bandwidth = estimate_bandwidth(boxes, quantile=0.2, n_samples=500)
        else:
            bandwidth = self.config.CLUST_BW
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(boxes)
        cluster_labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        cluster_idx_per_object = {i: {'boxes':[], 'scores':[]} for i in range(cluster_centers.shape[0])}
        for idx, c in enumerate(cluster_labels):
            cluster_idx_per_object[c]['boxes'].append(idx)
            cluster_idx_per_object[c]['scores'].append(scores[idx])
        keep_list = []
        for cluster_id, cluster in cluster_idx_per_object.items():
            max_score = np.argmax(cluster['scores'])
            keep_list.append(cluster['boxes'][max_score])

        keep = torch.Tensor(keep_list).to(labels)
        return boxes[keep], scores[keep], labels[keep]



    def prepare_predictions(self, class_logits, box_regression, proposals, image_shapes, scores_proposals):
        boxes_per_image = [boxes_in_image.shape[0]
                           for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # pred_scores = F.softmax(class_logits, -1)
        pred_scores = class_logits
        if scores_proposals is not None:
            scores_proposals = torch.cat(scores_proposals)
            pred_scores = torch.pow(class_logits, self.config.SCORE_MIXING_CLS) * \
                torch.pow(scores_proposals.unsqueeze(-1), self.config.SCORE_MIXING_LOC) #* \
                # torch.pow(iou_pred, 1.0)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        return pred_boxes_list, pred_scores_list
    
    def prepare_boxes(self, boxes, scores, image_shape, num_classes, device):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        assert self.nb_prototypes <= self.config.N_WAYS_TRAIN, 'Attempt to detect object from too many classes compared to training.'

        # Switch 1 / N+1 boxes output
        # boxes = boxes[:, 1].unsqueeze(1).repeat(1,max(self.config.N_WAYS_TRAIN, self.config.N_WAYS_TEST), 1)
        boxes = boxes[:, 1:]
        
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        return boxes, scores, labels