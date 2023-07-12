import torch
import cv2
import sys
import numpy as np
import copy
from collections import defaultdict
import matplotlib.pyplot as plt

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from .utils_map import *
from .multiclass_map.detection_map import DetectionMAP
from ..data.bbox import BBoxMode
from ..data.utils import filter_annotations
from ..data.dataset_categories import *
from ..config import device, mtqdm, NoStdStreams


'''
Wrapper around pycocotools' COCOeval class to match with the way 
we load data and we get predictions.
'''
class EvaluatorCOCO():
    def __init__(self, dataloader, model, n_class):
        self.dataloader = dataloader
        self.model = model.to(device)

        self.img_ids = []
        self.eval_imgs =[]

        self.dataset_name, self.dataset_modes = get_dataloader_infos(
            self.dataloader, eval_mode=True)
        
        # if n_class and nb_class_predictor are equals it means that the 
        # model was trained without a BG class, hence predictions must be shifted
        nb_class_predictor = self.model.model.roi_heads.box_predictor.cls_score.out_features
        self.shift_target = n_class == nb_class_predictor 

    
    def eval(self, prediction_mode=[BBoxMode.XYXY, BBoxMode.ABS], printing_on=False):
        self.model.eval()
        out_stream, err_stream = (sys.stdout, sys.stderr) if printing_on else (None, None)
        with NoStdStreams(out_stream, err_stream):
            dataset_coco_format = convert_to_coco_api(self.dataloader.dataset)

            self.evaluator_coco = COCOeval(dataset_coco_format, iouType='bbox')
            for idx, (image_tensor, \
                bbox_tensor_list, \
                label_tensor_list, \
                keep_list, \
                    indices) in enumerate(mtqdm(self.dataloader, desc='COCO evaluation')):

                bbox_tensor_list, label_tensor_list = filter_annotations(bbox_tensor_list, label_tensor_list, keep_list, mode='task_labels')

                # in case data augmentation (affine transform) removes all boxes from image
                if 0 in [label.shape[0] for label in label_tensor_list]:
                    continue
                predictions = self.model(image_tensor.to(device))
                # print(predictions)

                predictions = [{k: (v.cpu() + 1 if k == 'labels' and self.shift_target else v.cpu())
                            for k, v in t.items()}
                            for t in predictions]
                
                
                res = {id_: pred for id_,
                    pred in zip(indices, predictions)}
                
                # Test if metric behaves correctly       
                # res_ = self.get_fake_res(bbox_tensor_list, indices, label_tensor_list)

                img_ids = list(np.unique(list(res.keys())))
                self.img_ids.extend(img_ids)
                
                res_coco_format = self.prepare(res, 
                                                pred_modes=prediction_mode, 
                                                target_modes=self.dataset_modes,
                                                image_size=torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2))
                # print(res_coco_format)
                coco_dt = loadRes(dataset_coco_format,
                                res_coco_format) if res_coco_format else COCO()

                self.evaluator_coco.cocoDt = coco_dt
                self.evaluator_coco.params.imgIds = list(img_ids)

                img_ids, eval_imgs = evaluate(self.evaluator_coco)
                
                self.eval_imgs.append(eval_imgs)

                # if idx > 5:
                #     break
            
            self.eval_imgs = np.concatenate(self.eval_imgs, 2)
            create_common_coco_eval(self.evaluator_coco, self.img_ids, self.eval_imgs)
            self.evaluator_coco.accumulate()
            self.evaluator_coco.summarize()
            
            torch.cuda.empty_cache()
            return self.evaluator_coco.stats[0]
    
    def get_fake_res(self, targets, indices, labels):
        fake_pred = [{'boxes':target, 'labels': label, 'scores': torch.ones(label.shape[0])} for target, label in zip(targets, labels)]
        return {id_: pred for id_, pred in zip(indices, fake_pred)}
    
    def prepare(self, predictions, pred_modes=[BBoxMode.XYXY, BBoxMode.ABS], target_modes=[BBoxMode.XYXY, BBoxMode.ABS], image_sizes=None):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            

            if pred_modes[0] == BBoxMode.XYXY:
                boxes = convert_to_xywh(boxes)

            # if pred_modes[1] != target_modes[1] and target_modes[1] == BBoxMode.REL:
            #     boxes = boxes / image_size
            # elif pred_modes[1] != target_modes[1]:
            #     boxes = boxes * image_size

            # print('p',pred_modes)
            # print(target_modes)
        
            if pred_modes[1] == BBoxMode.REL:
                boxes = boxes * image_sizes
            
            boxes = boxes.tolist()
            # print(boxes)
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist() # labels 0 is for bg class and was removed 

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


'''
Wrapper around VOC mAP computation class from 
https://github.com/MathGaron/mean_average_precision
to match with the way we load data and we get predictions.

The code is under multiclass_map folder.
'''
class EvaluatorVOC():
    '''
    Needs to get targets and predictions boxes as [x1 y1 x2 y2] not [x1 y1 w h]
    '''
    def __init__(self, dataloader, model, n_class, few_shot_classes=None):
        self.dataloader = dataloader
        self.model = model.to(device)

        self.img_ids = []
        self.eval_imgs = []

        self.few_shot_classes = few_shot_classes
        self.mAP = DetectionMAP(n_class)

        self.dataset_name, self.dataset_modes = get_dataloader_infos(
            self.dataloader)
        

    def eval(self, prediction_mode=[BBoxMode.XYXY, BBoxMode.ABS], printing_on=False, verbose=True):
        self.model.eval()
        for idx, (image_tensor,
                  bbox_tensor_list,
                  label_tensor_list,
                  keep_list,
                  indices) in (enumerate(mtqdm(self.dataloader, desc='VOC per class mAP computation')) if verbose \
                                else enumerate(self.dataloader)):

            bbox_tensor_list, label_tensor_list = filter_annotations(bbox_tensor_list, label_tensor_list, keep_list, mode='task_labels')
            if self.few_shot_classes is not None:
                labels_converted = []
                for labels in label_tensor_list:
                    few_shot_classes_tensor = torch.Tensor(self.few_shot_classes)
                    labels_converted.append(torch.where(few_shot_classes_tensor == labels.unsqueeze(-1))[1])
            
                label_tensor_list = labels_converted

            # in case data augmentation (affine transform) removes all boxes from image
            if 0 in [label.shape[0] for label in label_tensor_list]:
                continue
            predictions = self.model(image_tensor.to(device))

            if self.few_shot_classes is not None:
                predictions = [{k: v.cpu().detach() if k == 'labels' 
                                                            else v.cpu().detach()
                                for k, v in t.items()}
                            for t in predictions]
        
            else:
                predictions = [{k: v.cpu().detach() if k == 'labels' 
                                                            else v.cpu().detach()
                                for k, v in t.items()}
                            for t in predictions]

            
            predictions = prepare_predictions_boxes(predictions,
                                                    prediction_mode,
                                                    self.dataset_modes, 
                                                    torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2))

            bbox_tensor_list = prepare_targets_boxes(bbox_tensor_list,
                                                    self.dataset_modes,
                                                     torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2))

            
            frames = []
            # print([pred['labels'] for pred in predictions])
            # print('GT ')
            # print(label_tensor_list)
            for img_id, pred in enumerate(predictions):
                frames.append([pred['boxes'].numpy(),
                              pred['labels'].numpy() + 1,
                              pred['scores'].numpy(),
                              bbox_tensor_list[img_id].numpy(),
                              label_tensor_list[img_id].numpy() + 1])

            for frame in frames:
                self.mAP.evaluate(*frame)
            


        # classes_names = DOTA_CATEGORIES if self.dataset_name == 'DOTA' else None

        self.mAP.compute_ap_classe(class_names=None)
        average_ap = self.mAP.get_AP(return_map= not printing_on, verbose=verbose)
        torch.cuda.empty_cache()
        return average_ap

    # def get_fake_res(self, targets, labels):
    #     fake_pred = [{'boxes': target + 0.05 * torch.randn(target.size()), 'labels': label, 'scores': torch.rand(
    #         label.shape[0])} for target, label in zip(targets, labels)]
    #     return fake_pred
