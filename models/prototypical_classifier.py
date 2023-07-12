import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from ..config import device
from torchvision.models.detection.image_list import ImageList
from ..data.utils import *
from .prototypes_utils import *
from .poolers import MultiScaleRoIAlign
from .rpn import RPNHead

class PrototypicalRPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors, config, anchor_generator=None):
        super(PrototypicalRPNHead, self).__init__()
        self.config = config
        
        self.embedding_d = self.config.RPN_EMBEDDING_D
        self.n_ways = self.config.N_WAYS_TRAIN
        self.n_classes = self.config.N_CLASSES - 1

        self.loss_embedding = 0 
        self.ALPHA = 0.5
        
        roi_align_size = 7
        self.roi_align = MultiScaleRoIAlign(
            ['0', '1', '2', '3'], roi_align_size, 2)

        self.shared_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, 512, kernel_size=1, stride=1),
            nn.LeakyReLU())
        self.cls_logits = nn.Conv2d(
            512, 1 * self.embedding_d, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            512, num_anchors * 4, kernel_size=1, stride=1
        )


        self.prototypes_refiner = nn.Sequential(nn.Linear(roi_align_size*roi_align_size*self.embedding_d, 512),
                                                nn.LeakyReLU(inplace=False),
                                                # nn.Dropout(0.3),
                                                nn.Linear(512, self.embedding_d))
        
        self.sigmas = None
        self.sigmas_per_class = None
        self.sigma_scheduler = SigmaScheduler(sigma_min=0.001, sigma_max=0.1, N_iter=10000, N_ways=self.n_ways)

        # self.sigma_estimator = nn.Sequential(nn.Linear(roi_align_size*roi_align_size*self.embedding_d, 512),
        #                                      nn.ReLU(inplace=False),
        #                                     #  nn.Dropout(0.3),
        #                                      nn.Linear(512, 1),
        #                                     #  nn.Sigmoid())
        #                                     SigmaActivation())
        


        # self.prototype_pooler = nn.AvgPool2d(roi_align_size,1)
        
        self.prototypes_boxes = {}
        if anchor_generator is not None:
            self.anchor_generator = anchor_generator
        self.anchors = None

        # self.simplex = torch.from_numpy(SimplexGenerator(self.config.RPN_EMBEDDING_D).generate()).float()
        # self.prototypes = {i: [self.simplex[k + i * self.config.K_SHOTS].to(device) 
        #             for k in range(self.config.K_SHOTS)] for i in range(self.n_classes)}


    def forward(self, images, features):
        embeddings = []
        bbox_reg = []

        # discard first feature map because takes too much memory
        for feature in features:
            t = self.shared_convs(feature)
            embedding = self.cls_logits(t)
            embeddings.append(embedding)
            bbox_reg.append(self.bbox_pred(t))


        image_shapes = images.image_sizes
        embeddings_dict = {str(i): embeddings[i]
                           for i in range(len(embeddings))}

        self.anchors = self.anchor_generator(images, embeddings)

        pooled_embeddings, levels_embeddings = self.roi_align(
            embeddings_dict, self.anchors, image_shapes, return_levels=True)

        pooled_embeddings = self.prototypes_refiner(pooled_embeddings.flatten(start_dim=1))
        # pooled_embeddings = self.prototype_pooler(pooled_embeddings).view(-1, self.embedding_d)


        pooled_embeddings = pooled_embeddings / pooled_embeddings.norm(2, dim=-1).unsqueeze(1)

        probabilities, objectness = self.compute_probabilities(
            pooled_embeddings, len(image_shapes), levels_embeddings)
        
        return objectness, bbox_reg, self.anchors, probabilities, pooled_embeddings

    def forward_embeddings(self, images, features):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        embeddings = []
        for feature in features:
            t = self.shared_convs(feature)
            embedding = self.cls_logits(t)
            embeddings.append(embedding)
        
        image_shapes = images.image_sizes
        embeddings_dict = {str(i): embeddings[i]
                           for i in range(len(embeddings))}

        
        self.anchors = self.anchor_generator(images, embeddings)
        pooled_embeddings, levels_embeddings = self.roi_align(
            embeddings_dict, self.anchors, image_shapes, return_levels=True)

        pooled_embeddings = self.prototypes_refiner(
            pooled_embeddings.flatten(start_dim=1)) 
        # pooled_embeddings = self.prototype_pooler(pooled_embeddings).view(-1, self.embedding_d)

        pooled_embeddings = pooled_embeddings / pooled_embeddings.norm(p=2, dim=-1, keepdim=True)

        probabilities, objectness = self.compute_probabilities(
            pooled_embeddings, len(image_shapes), levels_embeddings)
        return pooled_embeddings, self.anchors, objectness


    def compute_embeddinds_from_boxes(self, images, features, boxes):
        embeddings = []
        for feature in features:
            t = self.shared_convs(feature)
            embedding = self.cls_logits(t)
            embeddings.append(embedding)
        
        image_shapes = images.image_sizes
        embeddings_dict = {str(i): embeddings[i]
                           for i in range(len(embeddings))}

        pooled_embeddings, levels_embeddings = self.roi_align(
            embeddings_dict, boxes, image_shapes, return_levels=True)

        # pooled_embeddings = self.prototype_pooler(pooled_embeddings).view(-1, self.embedding_d)
        pooled_embeddings = self.prototypes_refiner(
            pooled_embeddings.flatten(start_dim=1))

        pooled_embeddings = pooled_embeddings / pooled_embeddings.norm(p=2, dim=-1, keepdim=True)
        return pooled_embeddings


    def compute_distances(self, embeddings, N):
        distances = {}
        embeddings = embeddings.view(N, -1, 1, self.embedding_d)

        prototypes_count = [len(p) for c, p in self.prototypes.items() if self.is_proto(c)]

        if sum(prototypes_count) == len(prototypes_count):
            # when 1 example per prototype
            prototypes_stacked = torch.stack([p[0] for c, p in self.prototypes.items() if self.is_proto(c)]) \
                .view(1, 1, -1, self.embedding_d)
            self.distances_stacked = torch.norm(
                embeddings - prototypes_stacked, dim=-1) ** 2
        else:
            # when multiple examples are available for each prototype
            for c, prototypes in self.prototypes.items():
                if prototypes != []:
                    prototypes_stacked = torch.stack(prototypes).view(
                        1, 1, -1, self.embedding_d)
                    distances[c] = torch.norm(
                        embeddings - prototypes_stacked, dim=-1) ** 2
                    distances[c] = distances[c].min(dim=-1)[0]
            self.distances_stacked = torch.stack(
                [distances[c] for c in distances.keys()], dim=-1)

    def compute_probabilities(self, embeddings, N, levels_embeddings):

        self.sigmas = compute_sigmas(self.prototypes,
                                    self.sigmas_per_class,
                                    embeddings,
                                    self.get_nb_proto(),
                                    self.sigma_scheduler, 
                                    self.config,
                                    self.sigmas, 
                                    is_rpn=False,
                                    mode=self.config.RPN_SIGMA_MODE)

        self.compute_distances(embeddings, N)
        probabilities = torch.exp(-self.distances_stacked / 2 / self.sigmas**2)

        if self.config.USE_PRIOR:
            priors = self.get_priors(self.config.DATASET_META.prior)
            priors = priors.to(probabilities).unsqueeze(0).unsqueeze(0)
            probabilities = probabilities * priors


        objectness = probabilities.max(dim=-1)[0]
        return probabilities, objectness

    def build_prototypes(self, support_set, backbone):
        self.prototypes = {i: [] for i in range(self.n_classes)}
        self.prototypes_boxes = {i: [] for i in range(self.n_classes)}
        self.prototypes_levels = {i: [] for i in range(self.n_classes)}
        self.sigmas_per_class = {i: [] for i in range(self.n_classes)}

        self.prototypes_features = []
        self.current_classes = torch.unique(support_set.target_allowed)


        support_loader = support_set.get_dataloaders(batch_size=1, shuffle=False)

        for image_tensor, boxes_list, labels_list, keep_list, indices in support_loader:
            
            # boxes_list, labels_list = filter_annotations(boxes_list, labels_list, keep_list, mode='task_labels')
            features = list(backbone(image_tensor.to(device)).values())
            
            image_shapes = [(t.shape[-2], t.shape[-1]) for t in image_tensor]
            img_list = ImageList(image_tensor, image_shapes)

            if self.anchors is None:
                self.anchors = self.anchor_generator(img_list, features)

            boxes_scale_tensor = torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2).to(device)

            if self.config.EXACT_BOXES:
                selected_boxes = [boxes.to(device)*boxes_scale_tensor for boxes in boxes_list]
            else:
                selected_boxes = select_boxes([boxes.to(device)*boxes_scale_tensor for boxes in boxes_list], self.anchors)
            

            selected_boxes = box_enlarger(selected_boxes, min_size=self.config.MIN_BOX_SIZE)
            labels = torch.cat(labels_list)
            self.prototypes_features.append(([feat.cpu() for feat in features], selected_boxes, img_list, labels)) # .clone()

            embeddings = []
            for feature in features:
                t = self.shared_convs(feature)
                logits = self.cls_logits(t)
                embeddings.append(logits)

            embeddings = {str(i): embeddings[i]
                        for i in range(len(embeddings))}

            pooled_embeddings, levels_embeddings = self.roi_align(
                    embeddings, selected_boxes, image_shapes, return_levels=True)

            pooled_embeddings = pooled_embeddings.flatten(start_dim=1)
            # sigma_predicted = self.sigma_estimator(pooled_embeddings)
            pooled_embeddings_refined = self.prototypes_refiner(
                pooled_embeddings)

            # pooled_embeddings = self.prototype_pooler(pooled_embeddings).view(-1,self.embedding_d)
            sigma_predicted = torch.ones(1,10)


            prototypes_boxes = torch.stack([box for boxes in boxes_list for box in boxes])

            for batch_idx, (label, proto) in enumerate(zip(labels, pooled_embeddings_refined)):
                idx = label.item()

                # moving average on the prototypes
                # alpha = self.config.ALPHA
                # alpha = 1.0
                # if not self.training:
                #     alpha = 1.0
                # proto_normed = proto / proto.norm(dim=-1, keepdim=True)
                # old_proto = self.prototypes[idx]
                # distance_to_old = torch.sum((torch.stack(old_proto) - proto.unsqueeze(0))**2, dim=-1)
                # closest_idx = distance_to_old.argmin().item()
                # closest_proto = old_proto[closest_idx]

                # new_proto = (1 - alpha) * closest_proto + alpha * proto_normed
                # self.prototypes[idx][closest_idx] = new_proto / new_proto.norm(dim=-1, keepdim=True)
                self.prototypes[idx].append(proto)

                self.prototypes_boxes[idx].append(
                    prototypes_boxes[batch_idx].to(device))
                self.prototypes_levels[idx].append(
                    levels_embeddings[batch_idx])
                self.sigmas_per_class[idx].append(
                    sigma_predicted[batch_idx])


        if self.config.RPN_PROTO_MODE != 'all':
            for c, protos in self.prototypes.items():
                if protos != [] and (self.config.RPN_PROTO_MODE == 'mean' or self.config.RPN_PROTO_MODE[c] =='mean'):
                    self.prototypes[c] = [torch.mean(torch.stack(protos, dim=0), dim=0)]

        for c, protos in self.prototypes.items():
            for id_proto, p in enumerate(protos):
                if p.shape[0] > 0:
                    self.prototypes[c][id_proto] = (p / p.norm(p=2, dim = -1))


    def is_proto(self, c):
        return c in self.current_classes

    def get_nb_proto(self):
        return len(self.get_classes_episode())

    def get_classes_episode(self):
        classes_ep = [k for k,v  in self.prototypes.items() if v != []]
        classes_ep = self.current_classes.tolist()
        return classes_ep
    
    def get_priors(self, prior_dict):
        classes_episode = self.get_classes_episode()
        prior = torch.Tensor([prior_dict[c] for c in classes_episode])
        return prior


class PrototypicalFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_anchors, config, rpn_classifier, head):
        super(PrototypicalFastRCNNPredictor, self).__init__()
        self.config = config
        self.rpn_classifier = rpn_classifier
        self.n_classes = config.N_CLASSES - 1 
        self.n_ways = config.N_WAYS_TRAIN
        self.embedding_d = config.ROI_HEAD_EMBEDDING_D
        self.head = head

        # self.cls_score = nn.Linear(in_channels, self.embedding_d)
        self.bbox_pred = nn.Sequential( 
                                        nn.Linear(in_channels, in_channels),
                                        nn.LeakyReLU(),
                                        nn.Linear(in_channels, 4 * (self.n_ways + 1)))
                                        # nn.Linear(in_channels, 4 ))

        self.cls_score = nn.Sequential(
                                        nn.Linear(in_channels, in_channels),
                                        nn.LeakyReLU(),
                                        nn.Linear(in_channels, self.embedding_d))

        self.sigmas = None
        self.sigma_scheduler = SigmaScheduler(sigma_min=0.1, sigma_max=0.5, N_iter=10000, N_ways=self.n_ways)
        self.sigma_estimator = nn.Sequential(nn.Linear(in_channels, 512),
                                             nn.LeakyReLU(inplace=False),
                                            #  nn.Dropout(0.3),
                                             nn.Linear(512, 1),
                                             #  nn.Sigmoid())
                                            SigmaActivation())

        self.reconstruction_mlp = nn.Sequential(nn.Linear(self.embedding_d, 512),
                                             nn.LeakyReLU(inplace=False),
                                             nn.Linear(512, 7*7*256))
        

        self.nb_iter = 0
        self.prototypes = {i: [randn_normed(self.config.ROI_HEAD_EMBEDDING_D).to(device) 
                            for k in range(self.config.K_SHOTS)] for i in range(self.n_classes)}
        


    def forward(self, x, levels_embeddings):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        embeddings = self.cls_score(x)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # Switch 1 / N+1 boxes output
        # bbox_deltas = self.bbox_pred(x).repeat(1, self.config.N_WAYS_TRAIN+1)
        bbox_deltas = self.bbox_pred(x)

        scores = self.compute_probabilities(embeddings, levels_embeddings)
        reconstructions = self.reconstruction_mlp(embeddings)
        return scores, bbox_deltas, embeddings, reconstructions
    
    def compute_distances(self, embeddings):
        distances = {k: torch.Tensor([])
                     for k, v in self.prototypes.items() if self.is_proto(k)}

        prototypes_count = [len(p)
                            for c,p in self.prototypes.items() if self.is_proto(c)]
        embeddings = embeddings.unsqueeze(1)

        if sum(prototypes_count) == len(prototypes_count):
            # when 1 exemple per prototype
            prototypes_stacked = torch.stack([p[0] for c, p in self.prototypes.items() if self.is_proto(c)]) \
                .view(1, -1, self.embedding_d)
            self.distances_stacked = torch.norm(
                embeddings - prototypes_stacked, dim=-1) ** 2
            
            if self.training:
                self.nb_iter += 1
                if not self.config.DEBUG:
                    plot_debug_hist(embeddings, 
                                prototypes_stacked, 
                                list([k for k,v in self.prototypes.items() if self.is_proto(k)]), 
                                self.distances_stacked, 
                                self.config.logger, self.nb_iter)

        else:
            # when multiple examples are available for each prototype
            for c, prototypes in self.prototypes.items():
                if self.is_proto(c):
                    prototypes_stacked = torch.stack(
                        prototypes).view(1, -1, self.embedding_d)
                    distances[c] = torch.norm(
                        embeddings - prototypes_stacked, dim=-1) ** 2
                    distances[c] = distances[c].min(dim=-1)[0]
            self.distances_stacked = torch.stack(
                [distances[c] for c in distances.keys()], dim=-1)



    def compute_probabilities(self, embeddings, levels_embeddings):
        self.compute_distances(embeddings)
        
        self.sigmas = compute_sigmas(self.prototypes,
                                    self.sigmas_per_class,
                                    embeddings,
                                    self.get_nb_proto(),
                                    self.sigma_scheduler, 
                                    self.config, 
                                    self.sigmas, 
                                    is_rpn=False,
                                    mode=self.config.ROI_HEAD_SIGMA_MODE)

        proba_classes = torch.exp(-self.distances_stacked / 2 / self.sigmas ** 2)

        if self.config.USE_PRIOR:
            priors = self.get_priors(self.config.DATASET_META.prior)
            priors = priors.to(proba_classes).unsqueeze(0)
            proba_classes = proba_classes * priors
            

        try:
            proba_bg = 1 - proba_classes.max(dim=-1, keepdim=True)[0]
        except:
            print(proba_classes.shape)
        probabilities = torch.cat([proba_bg, proba_classes], dim=-1)
        # probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return probabilities

    def build_prototypes(self):
        # self.prototypes = {i: [] for i in range(self.n_classes)}
        for c, proto_list in self.prototypes.items():
            self.prototypes[c] = [p.detach() for p in proto_list]
        self.sigmas_per_class = {i: [] for i in range(self.n_classes)}
        self.prototypes_levels = {i: [] for i in range(self.n_classes)}

        for features, boxes, img_list,  labels in self.rpn_classifier.prototypes_features:
            features = [feat.to(device) for feat in features]
            pooled_feat, levels_prototypes = self.rpn_classifier.roi_align({str(i):features[i] for i in range(len(features))},
                                            boxes, img_list.image_sizes, return_levels=True)
            pooled_feat = self.head(pooled_feat)
            pooled_feat_ = self.cls_score(pooled_feat)
            sigma_predicted = self.sigma_estimator(pooled_feat)

            for proto, sigma, c, level in zip(pooled_feat_, sigma_predicted, labels, levels_prototypes):
                # Moving average prototypes

                alpha = self.config.ALPHA
                if not self.training:
                    alpha = 1.0

                proto_normed = proto / proto.norm(dim=-1, keepdim=True)
                old_proto = self.prototypes[c.item()]
                distance_to_old = torch.sum((torch.stack(old_proto) - proto.unsqueeze(0))**2, dim=-1)
                closest_idx = distance_to_old.argmin().item()
                closest_proto = old_proto[closest_idx]

                new_proto = (1 - alpha) * closest_proto + alpha * proto_normed
                self.prototypes[c.item()][closest_idx] = new_proto / new_proto.norm(dim=-1, keepdim=True)
                # self.prototypes[c.item()].append(proto)
                self.sigmas_per_class[c.item()].append(sigma)
                self.prototypes_levels[c.item()].append(level)
        

        if self.config.ROI_PROTO_MODE != 'all':
            for c, protos in self.prototypes.items():
                if self.is_proto(c) and (self.config.RPN_PROTO_MODE == 'mean' or self.config.RPN_PROTO_MODE[c] =='mean'):
                    self.prototypes[c] = [torch.mean(torch.stack(protos, dim=0), dim=0)]

        for c, protos in self.prototypes.items():
            for id_proto, p in enumerate(protos):
                if p.shape[0] > 0:
                    self.prototypes[c][id_proto] = (p / p.norm(p=2, dim = -1))
        

    def get_nb_proto(self):
        return len(self.get_classes_episode())

    def get_classes_episode(self):
        return self.rpn_classifier.get_classes_episode()
    
    def is_proto(self, classe):
        return classe in self.get_classes_episode()

    def get_priors(self, prior_dict):
        classes_episode = self.get_classes_episode()
        prior = torch.Tensor([prior_dict[c] for c in classes_episode])
        return F.softmax(prior / prior.max(), dim=0)


class NonPrototypicalRPNHead(RPNHead):
    def __init__(self, in_channels, num_anchors, config, anchor_generator):
        super().__init__(in_channels, num_anchors)

        self.config = config

        roi_align_size = 7
        self.roi_align = MultiScaleRoIAlign(
            ['0', '1', '2', '3'], roi_align_size, 2)
        
        self.anchor_generator = anchor_generator
        self.anchors = None
        self.distances_stacked = None
        self.sigmas = None
        
        

    def forward(self, images, features):
        objectness, bbox_pred = super().forward(features)

        if self.anchors is None:
            self.anchors = self.anchor_generator(images, features)
        objectness = torch.cat([obj.flatten(start_dim=1) for obj in objectness], dim=-1)
        return objectness, bbox_pred, self.anchors, objectness.unsqueeze(-1).repeat(1,1,3), torch.randn(21824,128).cuda()

    def build_prototypes(self, support_set, backbone):
        self.prototypes_features = []
        self.current_classes = torch.unique(support_set.target_allowed)

        support_loader = support_set.get_dataloaders(batch_size=1, shuffle=False)

        for image_tensor, boxes_list, labels_list, keep_list, indices in support_loader:
            features = list(backbone(image_tensor.to(device)).values())
            
            image_shapes = [(t.shape[-2], t.shape[-1]) for t in image_tensor]
            img_list = ImageList(image_tensor, image_shapes)

            boxes_scale_tensor = torch.tensor(image_tensor.shape[-2:][::-1]).repeat(2).to(device)
            selected_boxes = [boxes.to(device)*boxes_scale_tensor for boxes in boxes_list]
            
            selected_boxes = box_enlarger(selected_boxes, min_size=self.config.MIN_BOX_SIZE)
            labels = torch.cat(labels_list)
            self.prototypes_features.append(([feat.cpu() for feat in features], selected_boxes, img_list, labels)) # .clone()

            # embeddings = []
            # for feature in features:
            #     t = self.shared_convs(feature)
            #     logits = self.cls_logits(t)
            #     embeddings.append(logits)

            # embeddings = {str(i): embeddings[i]
            #             for i in range(len(embeddings))}

            # pooled_embeddings, levels_embeddings = self.roi_align(
            #         embeddings, selected_boxes, image_shapes, return_levels=True)

            # pooled_embeddings = pooled_embeddings.flatten(start_dim=1)
            # pooled_embeddings_refined = self.prototypes_refiner(
            #     pooled_embeddings)


            # for batch_idx, (label, proto) in enumerate(zip(labels, pooled_embeddings_refined)):
            #     idx = label.item()

            #     self.prototypes[idx].append(proto)




        # if self.config.RPN_PROTO_MODE != 'all':
        #     for c, protos in self.prototypes.items():
        #         if protos != [] and (self.config.RPN_PROTO_MODE == 'mean' or self.config.RPN_PROTO_MODE[c] =='mean'):
        #             self.prototypes[c] = [torch.mean(torch.stack(protos, dim=0), dim=0)]

        # for c, protos in self.prototypes.items():
        #     for id_proto, p in enumerate(protos):
        #         if p.shape[0] > 0:
        #             self.prototypes[c][id_proto] = (p / p.norm(p=2, dim = -1))


    def is_proto(self, c):
        return c in self.current_classes

    def get_nb_proto(self):
        return len(self.get_classes_episode())

    def get_classes_episode(self):
        classes_ep = self.current_classes.tolist()
        return classes_ep
    
    def get_priors(self, prior_dict):
        classes_episode = self.get_classes_episode()
        prior = torch.Tensor([prior_dict[c] for c in classes_episode])
        return prior