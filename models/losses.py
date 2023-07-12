import torch
import torch.nn as nn 
import torch.nn.functional as F
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss

from ..config import device


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha]).to(device)
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha).to(device)
        
    
    def forward(self, inputs, targets):
        assert (inputs.shape[-1] == self.alpha.shape[-1]) or (len(inputs.shape) == 1 and self.alpha.shape[-1] == 2), \
            'Using binary target with multiclass focal loss or vice-versa'

        # clamp probability value for numerical stability 
        inputs = torch.clamp(inputs, 1e-4, 1.0 - 1e-4)
        alpha = self.alpha.gather(0, targets.long())

        if len(inputs.shape) == 1:
            logpt = F.binary_cross_entropy(inputs, targets, weight=alpha, reduction='none')
        else:
            logpt = F.nll_loss(torch.log(inputs), targets, weight=self.alpha, reduction='none')
        
        pt = torch.exp(-logpt)
        focal_weight = torch.pow(1 - pt, self.gamma)
        cls_loss = focal_weight * logpt

        if self.reduction == 'mean':
            cls_loss = cls_loss.mean()

        return cls_loss

class EmbeddingLoss(nn.Module):
    def __init__(self, model):
        super(EmbeddingLoss, self).__init__()
        self.model = model

        self.triplet_loss = TripletMarginLoss(margin=0.5, triplets_per_anchor='all')

        self.nll_loss = torch.nn.CrossEntropyLoss()

        self.sim_loss = ProtoSimilarityLoss()
    
    def forward(self, inputs=None, targets=None, labels=None, indices=None, modes=[]):
        loss_value = 0
        if 'triplet' in modes:
            if indices is None:
                indices = torch.arange(inputs.shape[0]).to(inputs).long()
            if labels is None and targets is not None:
                labels = torch.where(
                    targets[indices].unsqueeze(-1) ==  \
                        torch.Tensor(self.model.get_classes_episode()).to(inputs))[-1] \
                            .view_as(targets[indices])
            # else:
            #     labels = labels.view_as(targets)
            loss_value += self.triplet_loss(inputs[indices], labels)
        if 'nll' in modes:
            loss_value += self.nll_loss(self.model.class_logits, self.model.class_logits_labels.long())
        if 'cos' in modes:
            prototypes_stacked = torch.stack([p for protos in list(self.model.prototypes.values()) for p in protos if protos != []])
            loss_value += 10 * self.sim_loss(prototypes_stacked) # times 10 to align with other loss values

        return loss_value

class ProtoSimilarityLoss(nn.Module):
    def __init__(self):
        super(ProtoSimilarityLoss, self).__init__()
        self.distance = CosineSimilarity()
        
    def forward(self, prototypes):
        similarities = self.distance(prototypes) ** 2
        
        def eye_like(tensor):
            return torch.eye(*tensor.size(), out=torch.empty_like(tensor))
        
        zero_diag = torch.ones_like(similarities)- eye_like(similarities)
        return 0.5 * torch.mean(similarities * zero_diag) 