import torch
import torch.nn.functional as F
import numpy as np
import time
import cv2
import torchvision.models.detection._utils as det_utils
import torchvision.ops.boxes as box_ops
from ..config import device
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SigmaEstimator():
    def __init__(self, means, sigmas, max_iter=10, mode='emp'):
        self.means = means.unsqueeze(-1)
        self.K = sigmas.shape[-1]
        self.d = means.shape[1]
        self.max_iter = max_iter
        self.sigmas = torch.stack([torch.eye(self.d).to(sigmas) * sigma for sigma in sigmas])
        self.pi_k = torch.log(torch.ones(self.K, 1).to(sigmas) / self.K)
        self.mode = mode


    def gaussian_pdf_all(self, x):
        normal = torch.distributions.MultivariateNormal(self.means[:,:,0], self.sigmas)
        return normal.log_prob(x.T.unsqueeze(1)).T

    def gamma_estimate(self, x):
        pdf = self.gaussian_pdf_all(x)
        num = pdf + self.pi_k
        return num - torch.logsumexp(num, dim=0, keepdim=True)
    
    def estimate_sigmas(self, x):
        if self.mode == 'em':
            return self.estimate_em(x)
        else:
            return self.estimate_classic(x)

    def estimate_em(self, x):
        '''
        x matrix of points with shape d, N
        '''
        with torch.no_grad():
            d, N = x.shape
            for i in range(self.max_iter):
                gamma = self.gamma_estimate(x)
                n_k = torch.logsumexp(gamma, dim=-1).unsqueeze(-1)
                self.pi_k = n_k - np.log(N)
                x_norm = (x - self.means).permute(0,2,1)
                logsig = - n_k + torch.logsumexp(gamma.unsqueeze(-1) +  torch.log(x_norm **2), dim=1)
                self.sigmas = torch.diag_embed(torch.exp(logsig), dim1=-2, dim2=-1)
            return torch.clamp(torch.diagonal(self.sigmas, dim1=1, dim2=2).sum(dim=-1), 0.05,0.5)
    
    def estimate_classic(self, x):
        # compute distances
        # assign labels to points 
        # group by label
        # estimate sigma on groups
        
        L2_dist = LpDistance()
        n_proto = self.means.shape[0]
        dist_matrix = L2_dist(self.means[:,:,0], x.T)
        labels = torch.argmax(dist_matrix, dim=0)
        sigmas = []
        for proto in range(n_proto):
            proto_ind = torch.where(labels == proto)[0]
            if proto_ind.shape[0] == 0:
                sigmas.append(0.1)
            else:
                sigmas.append(x.T[proto_ind].std())
        
        return torch.Tensor(sigmas).view(1,-1).to(device)



def compute_sigmas(prototypes, 
                    learned_sigmas, 
                    embeddings, 
                    n_proto, 
                    sigma_scheduler, 
                    config, 
                    old_sigmas, 
                    is_rpn=True, 
                    mode='fixed', 
                    verbose=False):
    '''
    Compute per prototype variance according to the selected mode.  
    
    Args:
        - prototypes: dict[list[Tensor]]
        - learned_sigmas: dict[list[Tensor]]
        - embeddings: embeddings vectors of boxes Nx256 
        - n_proto: number of prototypes currently used
        - sigma_scheduler: decay scheduler used to progressively reduce sigmas
        - conig: config object from RPN or ROI_HEAD
        - is_rpn: Bool that explicitly state if sigmas are computed in RPN or in ROI_HEAD
        - mode: computation mode 'learned' / 'estimated' / 'decay' / 'fixed'
        - verbose: when true display sigmas values
    
    Returns:
        - sigmas: Tensor
    '''
    embedding_d = config.RPN_EMBEDDING_D if is_rpn else config.ROI_HEAD_EMBEDDING_D

    if mode == 'learned':
        sigmas =  average_sigma_over_examples(
            learned_sigmas).view(1, -1)
        sigmas = torch.clamp(sigmas, 1e-3)
    
    elif mode == 'estimated':
        prototypes_stacked = torch.stack([p[0] for p in prototypes.values() if p != []]) \
            .view(-1, embedding_d)
        sigmas = torch.ones(n_proto).to(device) * 0.5
        estimator = SigmaEstimator(prototypes_stacked.detach(), sigmas)
        sigmas = estimator.estimate_sigmas(embeddings.detach().T)
        sigmas = sigmas.view(1, n_proto)
    
    elif mode == 'decay':
        sigmas = sigma_scheduler.decay(old_sigmas)
        
    elif mode == 'fixed':
        if type(config.ROI_HEAD_SIGMA) == float:
            sigmas = torch.ones(1, n_proto).to(device) * config.ROI_HEAD_SIGMA
        
        elif type(config.ROI_HEAD_SIGMA) == list:
            assert len(config.ROI_HEAD_SIGMA) == n_proto
            sigmas = torch.Tensor(config.ROI_HEAD_SIGMA).to(device).unsqueeze(0)
        
        elif type(config.RPN_SIGMA) == dict:
            sigmas_list = [s for class_id, s in config.RPN_SIGMA.items() if prototypes[class_id] != []]
            sigmas = torch.Tensor(sigmas_list).to(device).unsqueeze(0)
    
    if verbose:
        print(sigmas)

    return sigmas


def average_sigma_over_examples(sigmas):
    '''
    Averages sigmas over examples per class.

    Args:
        - sigmas dict[list[tensor]]
    ''' 
    # sigma_res = []
    # for c, sig in sigmas.items():
    #     if len(sig) > 0:
    #         sig_stacked = torch.cat(sig)
    #         sigma_res.append(torch.mean(sig_stacked))
    # return torch.cat(sigma_res)
    return torch.stack([torch.stack(sig).mean() for sig in sigmas.values() if len(sig)>0])


def pad_list(l, value, length):
    new_list = []
    i = 0
    while len(new_list) < length:
        if i < len(l):
            new_list.append(l[i])
        else:
            new_list.append(value)
        i += 1
    return new_list


def crop_img(img, box):
    x1, y1, x2, y2 = (
        box * torch.Tensor(list(img.shape[-2:])).repeat(2)).to(int)
    return img[..., y1:y2, x1:x2]


def pad_object(img, min_size_h=64, min_size_v=64):
    H, W = img.shape[-2:]
    h_pad = max(0, min_size_h - W)
    v_pad = max(0, min_size_v - H)
    left = h_pad // 2
    right = h_pad // 2 + h_pad % 2
    top = v_pad // 2
    bottom = v_pad // 2 + v_pad % 2
    size_h = max(min_size_h, W)
    size_v = max(min_size_v, H)
    box = torch.Tensor([top, left, size_v - bottom,
                        size_h - right]).type(img.dtype)
    padded_img = torch.nn.ConstantPad2d(
        (left, right, top, bottom), value=0)(img)
    return padded_img, box


def mask_img(img, box, value):
    x1, y1, x2, y2 = (
        box * torch.Tensor(list(img.shape[-2:])).repeat(2)).to(int)
    mask = torch.ones_like(img) * value
    mask[..., y1:y2, x1:x2] = img[..., y1:y2, x1:x2]
    # cv2.imwrite('test.jpg', mask[:,:,:].permute(
    #     1, 2, 0).cpu().numpy()*255)
    return mask


def select_boxes(boxes_list, anchors_list):
    selected_boxes = []

    for boxes, anchors in zip(boxes_list, anchors_list):
        match_quality_matrix = box_ops.box_iou(
            boxes.to(anchors), anchors)
        best_idx = match_quality_matrix.max(dim=0)[0].argmax()
        selected_boxes.append(anchors[best_idx].unsqueeze(0))
    return selected_boxes


def infer_scale(feature, original_size):
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    assert possible_scales[0] == possible_scales[1]
    return possible_scales[0]


def crop_pad_batch(images, boxes_list, labels_list, min_size=64):
    batch_img = []
    batch_boxes = []
    batch_labels = []

    max_box_size = [min_size, min_size]

    for batch_idx, boxes in enumerate(boxes_list):
        boxes = boxes * torch.Tensor(list(images.shape[-2:])).repeat(2)
        hw = boxes[:, 2:] - boxes[:, :2]
        max_hw = torch.ceil(hw.max(dim=0)[0]).to(int)

        max_box_size[0] = max(max_box_size[0], max_hw[0].item())
        max_box_size[1] = max(max_box_size[1], max_hw[1].item())

    for batch_idx, boxes in enumerate(boxes_list):
        for box, label in zip(boxes, labels_list[batch_idx]):
            cropped_img = crop_img(images[batch_idx], box)
            padded_img, padded_box = pad_object(
                cropped_img, *max_box_size)
            batch_img.append(padded_img)
            batch_boxes.append(padded_box)
            batch_labels.append(label)

    return torch.stack(batch_img), torch.stack(batch_boxes), torch.stack(batch_labels)

class SigmaActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, x):
        return torch.log(1 + torch.exp(x))

class SigmaScheduler():
    def __init__(self, sigma_min, sigma_max, N_iter, N_ways):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N_iter = N_iter
        self.sigma_ratio = np.power(sigma_min / sigma_max, 1/N_iter)
        self.sigmas = torch.ones(N_ways).to(device) * sigma_max
        
    
    def decay(self, sigmas):
        if sigmas is None:
            return self.sigmas
        elif type(sigmas) == list:
            return list(map(lambda s: self.decay_sigma(s), sigmas))
        else:
            return self.decay_sigma(sigmas)

    def decay_sigma(self, sigma):       
        return self.sigma_min + (sigma - self.sigma_min) * self.sigma_ratio
    
    def get_value_t(self, t):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * \
             np.power(self.sigma_ratio, t/self.n_iter)


def plot_debug_hist(embeddings, prototypes, labels, distances, logger, n_iter):
    sim = CosineSimilarity()
    distance_matrix = torch.cdist(embeddings[:,0,:], embeddings[:,0,:])
    pairwise_dist_emb = torch.sqrt(distance_matrix.flatten())

    proto_distance = torch.cdist(prototypes[0], prototypes[0])
    proto_sim = sim(prototypes[0])

    logger.writer.add_histogram('Pairwise distance', pairwise_dist_emb, n_iter, bins=100)
    
    for d_to_proto, label in zip(distances.T, labels):
        logger.writer.add_histogram('Distance to proto {}'.format(label), 
                                    d_to_proto, n_iter)

    logger.writer.add_image('Proto similarity', proto_sim.unsqueeze(0), n_iter)
    logger.writer.add_image('Proto distance', proto_distance.unsqueeze(0), n_iter)

def plot_best_proposals(proposals, targets, img, k=20):
    gt_boxes = [t["boxes"] for t in targets]
    gt_labels = [t["labels"] for t in targets]
    match_quality_matrix = box_ops.box_iou(gt_boxes[0], proposals[0])
    indices = torch.topk(match_quality_matrix.max(dim=0)[0], k).indices
    top_k_proposals = proposals[0][indices]

    fig,ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img.cpu().detach()[0].permute(1,2,0))
    scores = match_quality_matrix.max(dim=0)[0][indices]
    for prop, score in zip(top_k_proposals, scores):
        x, y, w, h = prop.cpu().detach().tolist() 
        patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(patch)
        ax.text(x, y-5, '{:.2f}'.format(score.item()), c='g')
    
    for box in gt_boxes[0]:
        x, y, w, h = box.cpu().detach().tolist() 
        patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(patch)
    
    plt.show()

def randn_normed(shape, dim=0):
    vect = torch.randn(shape)
    vect = vect / vect.norm(dim=dim, keepdim=True)
    return vect


class SimplexGenerator():
    '''
    Object that generate randomly oriented simplex. 
    In n-d space, it returns n+1 vectors of dimension n and norm 1. 
    '''
    def __init__(self, n, seed=42):
        self.n = n
        self.rng = np.random.RandomState(seed)
    
    def generate(self):
        return self.generate_random_simplex()
    
    def rot_mat(self, i, theta):
        r = np.eye(self.n)
        if i < self.n-1:
            r[i:i+2, i:i+2] = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),  np.cos(theta)]])
        else:
            r[0, 0] = np.cos(theta)
            r[-1, 0] = -np.sin(theta)
            r[-1, -1] = np.cos(theta)
            r[0, -1] = np.sin(theta)
        return r

    def random_rotation(self):
        '''
        Generate a random n-dimensional rotation matrix that rotates vectors wrt all axis
        '''
        R = np.eye(self.n)
        for i in range(self.n):
            theta = self.rng.rand() * np.pi/2
            R = R @ self.rot_mat(i, theta)
        return R

    def generate_random_simplex(self):
        '''
        Return a randomly oriented simplex. 
        '''
        aligned_simplex = self.gen_simplex(self.n, 0, self.n, first=True)
        R = self.random_rotation()
        simplex = R @ aligned_simplex.T
        return simplex.T
    
    def gen_simplex(self, n, start, N, first=False):
        '''
        Return n+1 points of dimension n that defines a simplex
        TODO when python 3.10 is available replace this by pattern matching
        '''
        simplex = np.zeros((n,n))
        simplex[0,0] = np.sqrt(1 - start)
        if n > 1:
            simplex[1:,0] = (- 1 / N - start) * 1 / simplex[0,0]
            simplex[1:,1:] = self.gen_simplex(n-1, start + simplex[1,0] ** 2, N)

        if first:
            last = - simplex.sum(axis=0)
            simplex = np.vstack([simplex, last])
        return simplex


