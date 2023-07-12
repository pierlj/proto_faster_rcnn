import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytorch_metric_learning import losses, distances
from torchvision.models.detection.image_list import ImageList

from ..config import device
from ..data.augmentation import DetectionTransform
from ..data.utils import box_enlarger

class SimCLRTrainer():
    def __init__(self, config, model, n_boxes=10):
        self.config = config
        self.model = model
        self.n_boxes = n_boxes
        self.augmentation = DetectionTransform(config, scale=(0.7, 1.3), degrees=45)
        self.contrastive_loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        self.loss = torch.nn.CrossEntropyLoss()
        self.lambda_bt = 0.1

        off_d = torch.ones(config.RPN_EMBEDDING_D) - torch.eye(config.RPN_EMBEDDING_D)
        self.off_diagonal = (off_d * self.lambda_bt + torch.eye(config.RPN_EMBEDDING_D)).to(device)
    
    
    def train_step(self, image_tensor, boxes_list=None):
        image_tensor = image_tensor.to(device, non_blocking=True)

        # extract features
        features = self.model.backbone(image_tensor)  

        img_size = torch.tensor(image_tensor.shape[-2:]).repeat(2).unsqueeze(0).to(device)
        if boxes_list is not None:
            boxes_list = [boxes.to(device) * img_size for boxes in boxes_list]
            box_count = [boxes.shape[0] for boxes in boxes_list]
            
            # boxes_list = box_enlarger(boxes_list) 

            if self.config.DEBUG_PLOT:
                self.plot_debug(image_tensor[0], image_tensor[1], boxes_list[0], boxes_list[1])

            img_list = ImageList(
                    image_tensor, [(t.shape[-2], t.shape[-1]) for t in image_tensor])
        
            # compute embedding of the cropped boxes (done with ROI Align) 
            embeddings = self.model.rpn.head.compute_embeddinds_from_boxes(
                img_list, list(features.values()), boxes_list)

        if self.config.SSL_METHOD == 'simsiam':
            z_a = features['2'][::2].flatten(start_dim=1)
            z_b = features['2'][1::2].flatten(start_dim=1)

            z_a = self.model.projection_mlp(z_a)
            z_b = self.model.projection_mlp(z_b)

            p_a = self.model.prediction_mlp(z_a)
            p_b = self.model.prediction_mlp(z_b)

            def distance(p, z):
                return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

            return distance(p_a, z_b) / 2 + distance(p_b, z_a) / 2


        elif self.config.SSL_METHOD == 'barlow':
            z_A = embeddings[::2]
            z_B = embeddings[1::2]

            N, D = z_A.shape

            z_A_norm = (z_A - z_A.mean(dim=0, keepdim=True))\
                / z_A.std(dim=0, keepdim=True)
                
            z_B_norm = (z_B - z_B.mean(dim=0, keepdim=True))\
                / z_B.std(dim=0, keepdim=True)

            cross_cor_matrix = torch.mm(z_A_norm.T, z_B_norm) / N

            c_diff = (cross_cor_matrix - torch.eye(D, device=device)).pow(2)

            c_diff = c_diff * self.off_diagonal

            return c_diff.mean()
        
        elif self.config.SSL_METHOD == 'simclr':
            embeddings = embeddings / torch.norm(embeddings, dim=-1).unsqueeze(1)
            labels = torch.arange(embeddings.shape[0]).to(device) // 2

            # sim_loss_rpn = self.contrastive_loss(embeddings, labels)

            # TO DO change below to match cdist B M R dims format
            distance_mat = -torch.cdist(embeddings, embeddings)

            # print((z_image_rpn - z_image_prime_rpn).sum())

            #######################################################################
            # Code from: https://github.com/sthalles/SimCLR/blob/master/simclr.py #
            #######################################################################
            if random.random() < 0.0:
                plt.imshow(-distance_mat.cpu().detach().numpy(), vmin=0.0, vmax=1.0)
                plt.show()

            # if augmentation if too strong and makes all objects out of the image 
            
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

            mask = torch.eye(labels.shape[0], dtype=torch.bool)
            labels = labels[~mask].view(labels.shape[0], -1)
            distance_mat = distance_mat[~mask].view(distance_mat.shape[0], -1)

            # select and combine multiple positives
            positives = distance_mat[labels.bool()].view(labels.shape[0], -1)

            # select only the negatives
            negatives = distance_mat[~labels.bool()].view(distance_mat.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)

            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

            temperature = 1.0
            logits = logits / temperature

            sim_loss_rpn = self.loss(logits, labels)

            return sim_loss_rpn 

    def plot_debug(self, image, image_prime, boxes, boxes_prime):
        fig, ax = plt.subplots(1,3)
        selected_box = random.randint(0, boxes.shape[0]-1)
        
        std = np.array(self.config.DATASET_META.std)
        mean = np.array(self.config.DATASET_META.mean)

        x, y, x_, y_ = (boxes[selected_box].cpu().numpy()).astype(int)
        crop = image[:,y:y_, x:x_].cpu().permute(1,2,0).numpy()
        crop = crop * std + mean
        ax[0].imshow(crop)
        
        x, y, x_, y_ = (boxes_prime[selected_box].cpu().numpy()).astype(int)
        crop = image_prime[:,y:y_, x:x_].cpu().permute(1,2,0).numpy()
        crop = crop * std + mean
        ax[1].imshow(crop)

        ax[2].imshow(image_prime.cpu().permute(1,2,0).numpy() * std + mean)
        patch = patches.Rectangle((x,y),x_-x,y_-y,linewidth=1,edgecolor='r',facecolor='none')
        ax[2].add_patch(patch)

        plt.show()


