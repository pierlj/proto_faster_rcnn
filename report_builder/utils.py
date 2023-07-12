import torch
import time
import numpy as np
import random
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.image_list import ImageList
from MulticoreTSNE import MulticoreTSNE as TSNE
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance

from ..task_sampler.sampler import TaskSampler
from ..data.datasets_info import *
from ..eval.eval import EvaluatorCOCO, EvaluatorVOC
from ..config import device

matplotlib.use('TkAgg')

def build_embeddings(model, base_path, seed=42):
    print("Building embeddings representations...")
    tic = time.time()
    config = model.config
    config.N_QUERY_TRAIN = 10
    config.N_QUERY_TEST = 10

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.cuda().eval()

    classes = list(np.arange(config.N_CLASSES-1))
    sampler = TaskSampler(config.DATASET_META, classes, config)
    (Q_train, S_train, classes_train), (Q_val, S_val, classes_val) = sampler. \
            sample_train_val_tasks(config.N_WAYS_TRAIN, 
            config.N_WAYS_TEST,
            config.K_SHOTS,
            config.N_QUERY_TRAIN,
            config.N_QUERY_TEST, verbose=False)

    embeddings = []
    labels_embedding = []
    scores_embeddings = []
    device = 'cuda:0'
    prototypes = []

    with torch.no_grad():
        model.model.rpn.fg_bg_sampler.batch_size_per_image = 1024
        model.build_prototypes(S_train, model.model.backbone)
        prototypes_train = model.model.rpn.head.prototypes

        for img, boxes_list, labels_list, keep_list, ind in Q_train.get_dataloaders(batch_size=1):
            targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * img.shape[-1],
                    'labels': labels_tensor.to(device, non_blocking=True)} 
                    for boxes_tensor, labels_tensor in zip(boxes_list, labels_list)]
            
            features = model.model.backbone(img.cuda())
            
            img_list = ImageList(
                img, [(t.shape[-2], t.shape[-1]) for t in img])
            embedding, anchors, scores = model.model.rpn(img_list, features, output_embeddings=True)
            prototypes.append(model.model.rpn.head.prototypes)

            labels, matched_gt_boxes, matched_idxs = model.model.rpn.assign_targets_to_anchors(
                    anchors, targets)
        
            classification_targets = [t['labels'][idx.clamp(min=0)] for t, idx in zip(targets, matched_idxs)]
            gt_labels = torch.cat(classification_targets)
            sampled_pos_inds, sampled_neg_inds = model.model.rpn.fg_bg_sampler(labels)
            
            sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
            sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

            for neg_idx, pos_idx in enumerate(sampled_pos_inds):

                embeddings.append(embedding[pos_idx].cpu())
                labels_embedding.append(gt_labels[pos_idx].cpu().item())
                scores_embeddings.append(scores[0,pos_idx].cpu().item())
                if random.random() < 1.0:
                    embeddings.append(embedding[sampled_neg_inds[neg_idx]].cpu())
                    labels_embedding.append(-1)
                    scores_embeddings.append(scores[0,sampled_neg_inds[neg_idx]].cpu().item())
        
        model.build_prototypes(S_val, model.model.backbone)
        prototypes_val = model.model.rpn.head.prototypes
        for img, boxes_list, labels_list, keep_list, ind in Q_val.get_dataloaders(batch_size=1):
            targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * img.shape[-1],
                    'labels': labels_tensor.to(device, non_blocking=True)} 
                    for boxes_tensor, labels_tensor in zip(boxes_list, labels_list)]
            
            features = model.model.backbone(img.cuda())
            
            img_list = ImageList(
                img, [(t.shape[-2], t.shape[-1]) for t in img])
            embedding, anchors, scores = model.model.rpn(img_list, features, output_embeddings=True)

            prototypes.append(model.model.rpn.head.prototypes)

            labels, matched_gt_boxes, matched_idxs = model.model.rpn.assign_targets_to_anchors(
                    anchors, targets)
        
            classification_targets = [t['labels'][idx.clamp(min=0)] for t, idx in zip(targets, matched_idxs)]
            gt_labels = torch.cat(classification_targets)
            
            sampled_pos_inds, sampled_neg_inds = model.model.rpn.fg_bg_sampler(labels)
            sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
            sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

            for neg_idx, pos_idx in enumerate(sampled_pos_inds):

                embeddings.append(embedding[pos_idx].cpu())
                labels_embedding.append(gt_labels[pos_idx].cpu().item())
                scores_embeddings.append(scores[0,pos_idx].cpu().item())

    prototypes_train_nb = [len(protos) for protos in list(prototypes_train.values())]
    prototypes_train_stacked = torch.stack([p for protos in list(prototypes_train.values()) for p in protos if protos != []])
    prototypes_train_classes = [c for c, protos in prototypes_train.items() for p in protos if protos != []]

    prototypes_val_nb = [len(protos) for protos in list(prototypes_val.values())]
    prototypes_val_stacked = torch.stack([p for protos in list(prototypes_val.values()) for p in protos if protos != []])
    prototypes_val_classes = [c for c, protos in prototypes_val.items() for p in protos if protos != []]

    embeddings, labels_embedding = filter_embeddings(embeddings, labels_embedding)
    embed_proto = torch.cat([embeddings, prototypes_train_stacked.cpu(), prototypes_val_stacked.cpu()], dim=0)

    embeddings_2d = TSNE(n_jobs=4, perplexity=5).fit_transform(embed_proto)

    random.seed(155) # to get nice colors
    colors_list = ['g', 'r', 'b', 'y', 'cyan', 'purple', 'teal', 'pink', 'olive', 'peru']
    colors_list = colors_list + ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(config.N_CLASSES-10)]
    cmap_emb = matplotlib.colors.ListedColormap(['k'] + colors_list)

    nb_proto_train = prototypes_train_stacked.shape[0]
    nb_proto_val = prototypes_val_stacked.shape[0]

    bounds = np.linspace(-1, config.N_CLASSES -1, config.N_CLASSES+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, config.N_CLASSES+1)

    vis_x = embeddings_2d[:-(nb_proto_train + nb_proto_val), 0]
    vis_y = embeddings_2d[:-(nb_proto_train + nb_proto_val), 1]


    fig, ax = plt.subplots(figsize=(10,10))
    scatter = ax.scatter(vis_x, vis_y, c=labels_embedding, marker='+', cmap=cmap_emb, norm=norm)
    # scatter = ax.scatter(vis_x, vis_y, c=scores_embeddings, marker='+', cmap=viridis)
    legend1 = ax.legend(*scatter.legend_elements(num=None), title="All classes",ncol=3, loc='upper right')
    ax.add_artist(legend1)

    # display train protos
    vis_x_p = embeddings_2d[-(nb_proto_train + nb_proto_val):-nb_proto_val, 0]
    vis_y_p = embeddings_2d[-(nb_proto_train + nb_proto_val):-nb_proto_val, 1]
    scatter = ax.scatter(vis_x_p, vis_y_p, c=prototypes_train_classes, marker='o', s=200, cmap=cmap_emb, norm=norm)
    train_legend = scatter.legend_elements(num=None)
    # ax.add_artist(legend1)

    # display val protos
    vis_x_p = embeddings_2d[-nb_proto_val:, 0]
    vis_y_p = embeddings_2d[-nb_proto_val:, 1]
    scatter = ax.scatter(vis_x_p, vis_y_p, c=prototypes_val_classes , marker='s', s=200, cmap=cmap_emb, norm=norm)
    test_legend = scatter.legend_elements(num=None)

    train_test_legend = [l1 + l2 for l1, l2 in zip(train_legend, test_legend)]
    legend1 = ax.legend(*train_test_legend, loc='upper left', title="Train/Test classes")
    ax.add_artist(legend1)
    plt.title('TSNE representation of embeddings and prototypes')
    plt.savefig(os.path.join(base_path, 'fig/rpn_emb.png'))

    plt.figure()            
    plt.hist(np.array(scores_embeddings), bins=50)
    plt.title('Scores histogram outputed from RPN')
    plt.xlabel('Objectness score')
    plt.ylabel('# of instances')
    plt.savefig(os.path.join(base_path, 'fig/rpn_hist.png'))

    plt.figure()
    distance = CosineSimilarity()
    plt.imshow((distance(torch.cat([prototypes_train_stacked, prototypes_val_stacked]))**2 ).cpu().numpy())
    plt.title('Similarity matrix between prototypes in RoI Heads')
    plt.savefig(os.path.join(base_path, 'fig/rpn_proto.png'))

    ####################################

    embeddings = []
    labels_embedding = []
    scores_embeddings = []
    device = 'cuda:0'
    prototypes = []

    with torch.no_grad():
        model.build_prototypes(S_train, model.model.backbone)
        prototypes_2_train = model.model.roi_heads.box_predictor.prototypes

        for img, boxes_list, labels_list, keep_list, ind in Q_train.get_dataloaders(batch_size=1):
            targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * img.shape[-1],
                    'labels': labels_tensor.to(device, non_blocking=True)} 
                    for boxes_tensor, labels_tensor in zip(boxes_list, labels_list)]
            
            features = model.model.backbone(img.cuda())
            
            img_list = ImageList(
                img, [(t.shape[-2], t.shape[-1]) for t in img])
    #         embedding, anchors, scores = model.model.rpn(img_list, features, output_embeddings=True)
            proposals, losses, scores = model.model.rpn(img_list, features, output_scores=True)

            embedding, boxes, scores = model.model.roi_heads(features, proposals, img_list.image_sizes, output_embeddings=True, scores_proposals=scores)
            
            gt_boxes = [t["boxes"].to(float) for t in targets]
            gt_labels = [t["labels"] + 1 for t in targets]
            
            matched_idxs, labels = model.model.roi_heads.assign_targets_to_proposals(
                proposals, gt_boxes, gt_labels)

            # sample a fixed proportion of positive-negative proposals
            sampled_inds = model.model.roi_heads.subsample(labels)[0]
            for sampled_idx in sampled_inds:
                if labels[0][sampled_idx].cpu().item() > 0 or random.random() < 0.01: 
                    embeddings.append(embedding[sampled_idx].cpu())
                    labels_embedding.append(labels[0][sampled_idx].cpu().item()-1)
                    scores_embeddings.append(scores[0][sampled_idx].max().cpu().item())
        
        model.build_prototypes(S_val, model.model.backbone)
        prototypes_2_val = model.model.roi_heads.box_predictor.prototypes
        for img, boxes_list, labels_list, keep_list, ind in Q_val.get_dataloaders(batch_size=1):
            targets = [{'boxes': boxes_tensor.to(device, non_blocking=True) * img.shape[-1],
                    'labels': labels_tensor.to(device, non_blocking=True)} 
                    for boxes_tensor, labels_tensor in zip(boxes_list, labels_list)]
            
            features = model.model.backbone(img.cuda())
            
            img_list = ImageList(
                img, [(t.shape[-2], t.shape[-1]) for t in img])
    #         embedding, anchors, scores = model.model.rpn(img_list, features, output_embeddings=True)
            proposals, losses, scores = model.model.rpn(img_list, features, output_scores=True)
            embedding, boxes, scores = model.model.roi_heads(features, proposals, img_list.image_sizes, output_embeddings=True, scores_proposals=scores)
            
            gt_boxes = [t["boxes"].to(float) for t in targets]
            gt_labels = [t["labels"] + 1 for t in targets]
            
            matched_idxs, labels = model.model.roi_heads.assign_targets_to_proposals(
                proposals, gt_boxes, gt_labels)

            # sample a fixed proportion of positive-negative proposals
            sampled_inds = model.model.roi_heads.subsample(labels)[0]
            for sampled_idx in sampled_inds:
                if labels[0][sampled_idx].cpu().item() > 0 or random.random() < 0.01: 
                    embeddings.append(embedding[sampled_idx].cpu())
                    labels_embedding.append(labels[0][sampled_idx].cpu().item()-1)
                    scores_embeddings.append(scores[0][sampled_idx].max().cpu().item())

    prototypes_train_nb = [len(protos) for protos in list(prototypes_2_train.values())]
    prototypes_train_stacked = torch.stack([p for c, protos in prototypes_2_train.items() for p in protos if c in classes_train])
    prototypes_train_classes = [c for c, protos in prototypes_2_train.items() for p in protos if c in classes_train]

    prototypes_val_nb = [len(protos) for protos in list(prototypes_2_val.values())]
    prototypes_val_stacked = torch.stack([p for c, protos in prototypes_2_val.items() for p in protos if c in classes_val])
    prototypes_val_classes = [c for c, protos in prototypes_2_val.items() for p in protos if c in classes_val]

    embed_proto = torch.cat([torch.stack(embeddings), prototypes_train_stacked.cpu(), prototypes_val_stacked.cpu()], dim=0)
    embeddings_2d = TSNE(n_jobs=4, perplexity=5).fit_transform(embed_proto.detach())

    nb_proto_train = prototypes_train_stacked.shape[0]
    nb_proto_val = prototypes_val_stacked.shape[0]

    vis_x = embeddings_2d[:-(nb_proto_train + nb_proto_val), 0]
    vis_y = embeddings_2d[:-(nb_proto_train + nb_proto_val), 1]


    fig, ax = plt.subplots(figsize=(10,10))
    # scatter = ax.scatter(vis_x, vis_y, c =labels_embedding, marker='+')
    scatter = ax.scatter(vis_x, vis_y, c=labels_embedding, marker='+', cmap=cmap_emb, norm=norm)
    # scatter = ax.scatter(vis_x, vis_y, c=distances.numpy(), marker='+', cmap=viridis)
    legend1 = ax.legend(*scatter.legend_elements(num=None), title="All classes", ncol=3, loc='upper right')
    ax.add_artist(legend1)

    # display train protos
    vis_x_p = embeddings_2d[-(nb_proto_train + nb_proto_val):-nb_proto_val, 0]
    vis_y_p = embeddings_2d[-(nb_proto_train + nb_proto_val):-nb_proto_val, 1]
    scatter = ax.scatter(vis_x_p, vis_y_p, c=prototypes_train_classes, marker='o', s=200, cmap=cmap_emb, norm=norm)
    train_legend = scatter.legend_elements()
    # display val protos
    vis_x_p = embeddings_2d[-nb_proto_val:, 0]
    vis_y_p = embeddings_2d[-nb_proto_val:, 1]
    scatter = ax.scatter(vis_x_p, vis_y_p, c=prototypes_val_classes , marker='s', s=200, cmap=cmap_emb, norm=norm)
    test_legend = scatter.legend_elements()

    train_test_legend = [l1 + l2 for l1, l2 in zip(train_legend, test_legend)]
    legend1 = ax.legend(*train_test_legend, loc='upper left', title="Train/Test classes")
    plt.title('TSNE representation of embeddings and prototypes')
    plt.savefig(os.path.join(base_path, 'fig/roi_emb.png'))

    plt.figure()
    plt.hist(scores_embeddings, bins=50)
    plt.title('Scores histogram outputed from RoI Heads')
    plt.xlabel('Class confidence score')
    plt.ylabel('# of instances')
    plt.savefig(os.path.join(base_path, 'fig/roi_hist.png'))

    distance = CosineSimilarity()
    plt.figure()
    plt.imshow((distance(torch.cat([prototypes_train_stacked, prototypes_val_stacked]))**2).cpu().numpy())
    plt.title('Similarity matrix between prototypes in RoI Heads')
    plt.savefig(os.path.join(base_path, 'fig/roi_proto.png'))

    print('Done in {:.2f}s.'.format(time.time() - tic))





def build_proposals(model, base_path, seed=42):
    print('Building proposals figures')
    tic = time.time()
    model = model.cuda().eval()
    config=model.config

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    classes = list(np.arange(config.N_CLASSES-1))
    sampler = TaskSampler(config.DATASET_META, classes, config)
    (Q_train, S_train, classes_train), (Q_val, S_val, classes_val) = sampler. \
            sample_train_val_tasks(config.N_WAYS_TRAIN, 
            config.N_WAYS_TEST,
            config.K_SHOTS,
            config.N_QUERY_TRAIN,
            config.N_QUERY_TEST, verbose=False)
    
    img, bbox, label, keep, idinces = next(iter(Q_train.get_dataloaders(batch_size=2)))
    img = img.cuda()
    img_list = ImageList(img, [(t.shape[-2], t.shape[-1]) for t in img])
    with torch.no_grad():
        model.build_prototypes(S_train, model.model.backbone)

        features = model.model.backbone(img)
        proposals_train_train, _, scores_train_train = model.model.rpn(img_list, features, output_scores=True)
        proposals_val_train, _, scores_val_train = model.model.rpn(img_list, features, output_scores=True)
    img_train = img
    img, bbox, label, keep, idinces = next(iter(Q_val.get_dataloaders(batch_size=2)))
    img = img.cuda()

    img_list = ImageList(img, [(t.shape[-2], t.shape[-1]) for t in img])
    with torch.no_grad():
        model.build_prototypes(S_train, model.model.backbone)
        features = model.model.backbone(img)
        proposals_train_val, _, scores_train_val = model.model.rpn(img_list, features, output_scores=True)
        model.build_prototypes(S_val, model.model.backbone)
        proposals_val_val, _, scores_val_val = model.model.rpn(img_list, features, output_scores=True)
    img_val = img

    # Create figure and axes
    fig,axs = plt.subplots(len(proposals_train_train),6, figsize=(20,6), constrained_layout=True)

    for idx in range(len(proposals_train_train)):
        mean=config.DATASET_META.mean
        std=config.DATASET_META.std

        img_ = torch.clamp(img_train[idx].cpu().permute(1,2,0) * torch.tensor(std) + torch.tensor(mean), 0, 1)
        img_ = img_.numpy()
        # pred = predictor(img_*255)

        # Display the images
        axs[idx, 0].imshow(img_)
        axs[idx,0].axis('off')
        im = axs[idx, 1].imshow(img_)
        axs[idx,1].axis('off')
        axs[idx, 2].imshow(img_)
        axs[idx,2].axis('off')  
        
        img_ = torch.clamp(img_val[idx].cpu().permute(1,2,0) * torch.tensor(std) + torch.tensor(mean), 0, 1)
        img_ = img_.numpy()
        
        axs[idx, 3].imshow(img_)
        axs[idx,3].axis('off')
        axs[idx, 4].imshow(img_)
        axs[idx,4].axis('off')
        axs[idx, 5].imshow(img_)
        axs[idx,5].axis('off')
        
        viridis = cm.get_cmap('viridis')
        norm = plt.Normalize(0, 1)
        
    #     color = (scores_train_train[idx] / scores_train_train[idx].max()).cpu().numpy()
        color = scores_train_train[idx].cpu().numpy()
        for idx_boxe, box in reversed(list(enumerate(proposals_train_train[idx]))):
            x, y, w, h = box.cpu().detach().tolist() 

            patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor=viridis(norm(color[idx_boxe])),facecolor='none')
            if color[idx_boxe] > 0.0:
                axs[idx, 1].add_patch(patch)

            
        color = (scores_val_train[idx]).cpu().numpy()
        for idx_boxe, box in reversed(list(enumerate(proposals_val_train[idx]))):
            x, y, w, h = box.cpu().detach().tolist() 

            patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor=viridis(norm(color[idx_boxe])),facecolor='none')
            axs[idx, 2].add_patch(patch)
        
        color = (scores_train_val[idx]).cpu().numpy()
        for idx_boxe, box in reversed(list(enumerate(proposals_train_val[idx]))):
            x, y, w, h = box.cpu().detach().tolist() 
            patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor=viridis(norm(color[idx_boxe])),facecolor='none')
            axs[idx, 4].add_patch(patch)

        
        color = (scores_val_val[idx]).cpu().numpy()
        for idx_boxe, box in reversed(list(enumerate(proposals_val_val[idx]))):
            x, y, w, h = box.cpu().detach().tolist() 
            patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor=viridis(norm(color[idx_boxe])),facecolor='none')
            axs[idx, 5].add_patch(patch)
        fig.colorbar(im, ax=axs[idx,5], label='Objectness score')
        
        axs[0, 0].title.set_text('Original Train Image')
        axs[0, 1].title.set_text('Train prototype/ train img')
        axs[0, 2].title.set_text('Val prototype/ train img')
        axs[0, 3].title.set_text('Original Val Image')
        axs[0, 4].title.set_text('Train prototype/ val img')
        axs[0, 5].title.set_text('Val prototype/ val img')

    plt.savefig(os.path.join(base_path, 'fig/proposals_full.png'))

    ##########

    img, bbox, label, keep, indices = next(iter(Q_train.get_dataloaders(batch_size=2)))
    img = img.cuda()
    img_list = ImageList(img, [(t.shape[-2], t.shape[-1]) for t in img])
    allowed_classes = classes_train
    proposals_classes = []
    scores_classes = []

    with torch.no_grad():
        for selected_classe in allowed_classes:
            model.model.rpn.head.build_prototypes(S_train, model.model.backbone)

            model.model.rpn.head.prototypes = {c: v if c == selected_classe else [] for c,v in model.model.rpn.head.prototypes.items()}
            features = model.model.backbone(img)
            p_classe, _, s_classe = model.model.rpn(img_list, features, output_scores=True)
            proposals_classes.append(p_classe)
            scores_classes.append(s_classe)

    img_classes = img

    # Create figure and axes
    fig,axs = plt.subplots(len(proposals_classes[0]),len(allowed_classes) + 1, figsize=(20,8), constrained_layout=True)

    for idx in range(len(proposals_classes[0])):
        mean=config.DATASET_META.mean
        std=config.DATASET_META.std
        
        viridis = cm.get_cmap('viridis')
        norm = plt.Normalize(0, 1)
        
        img_ = torch.clamp(img_classes[idx].cpu().permute(1,2,0) * torch.tensor(std) + torch.tensor(mean), 0, 1)
        img_ = img_.numpy()

        for i in range(len(allowed_classes) + 1):
            # Display the images
            axs[idx, i].imshow(img_)
            axs[idx,i].axis('off')
            im = axs[idx, i].imshow(img_)
            if i>0:
                axs[0, i].title.set_text('Classe {}'.format(allowed_classes[i-1]))
                color = (scores_classes[i-1][idx]).cpu().numpy()
                for idx_boxe, box in reversed(list(enumerate(proposals_classes[i-1][idx]))):
                    x, y, w, h = box.cpu().detach().tolist() 

                    patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor=viridis(norm(color[idx_boxe])),facecolor='none')
                    axs[idx, i].add_patch(patch)
        
        fig.colorbar(im, ax=axs[idx, len(allowed_classes)], label='Objectness score')
        
        axs[0, 0].title.set_text('Original Image')

    plt.savefig(os.path.join(base_path, 'fig/proposals_train.png'))

    #############

    img, bbox, label, keep, indices = next(iter(Q_val.get_dataloaders(batch_size=2)))
    img = img.cuda()
    img_list = ImageList(img, [(t.shape[-2], t.shape[-1]) for t in img])
    allowed_classes = classes_val
    proposals_classes = []
    scores_classes = []

    with torch.no_grad():
        for selected_classe in allowed_classes:
            model.model.rpn.head.build_prototypes(S_val, model.model.backbone)
            model.model.rpn.head.prototypes = {c: v if c == selected_classe else [] for c,v in model.model.rpn.head.prototypes.items()}
            features = model.model.backbone(img)
            p_classe, _, s_classe = model.model.rpn(img_list, features, output_scores=True)
            proposals_classes.append(p_classe)
            scores_classes.append(s_classe)

    img_classes = img

    # Create figure and axes
    fig,axs = plt.subplots(len(proposals_classes[0]),len(allowed_classes) + 1, figsize=(20,8), constrained_layout=True)

    for idx in range(len(proposals_classes[0])):
        mean=config.DATASET_META.mean
        std=config.DATASET_META.std
        
        viridis = cm.get_cmap('viridis', 12)
        norm = plt.Normalize(0, 1)
        
        img_ = torch.clamp(img_classes[idx].cpu().permute(1,2,0) * torch.tensor(std) + torch.tensor(mean), 0, 1)
        img_ = img_.numpy()

        for i in range(len(allowed_classes) + 1):
            # Display the images
            axs[idx, i].imshow(img_)
            axs[idx,i].axis('off')
            im = axs[idx, i].imshow(img_)
            if i>0:
                axs[0, i].title.set_text('Classe {}'.format(allowed_classes[i-1]))
                color = (scores_classes[i-1][idx] / scores_classes[i-1][idx].max()).cpu().numpy()

                for idx_boxe, box in reversed(list(enumerate(proposals_classes[i-1][idx]))):
                    x, y, w, h = box.cpu().detach().tolist() 
                    patch = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor=viridis(norm(color[idx_boxe])),facecolor='none')
                    axs[idx, i].add_patch(patch)
        
        fig.colorbar(im, ax=axs[idx, len(allowed_classes)], label='Objectness score')
        
        axs[0, 0].title.set_text('Original Image')

    plt.savefig(os.path.join(base_path, 'fig/proposals_test.png'))
    print('Done in {:.2f}s.'.format(time.time() - tic))
        
    

def compute_metrics(model, shots=[1,3,5,10], num_val_episodes = 15, seed=42):
    print('Performing evaluation...')
    tic = time.time()

    metrics = {'shots' : shots,
            'train-train' : {s: [] for s in shots},
            'train-test' :  {s: [] for s in shots},
            'test-train' :  {s: [] for s in shots},
            'test-test' :  {s: [] for s in shots}}
    
    config = model.config
    model = model.to(device).eval()
    # meta_dict = {'train': vhr_meta_full, 'test':vhr_meta_full}
    meta_dict = {'train': dota_meta, 'test':dota_test_meta}
    classes = list(np.arange(config.N_CLASSES-1))
    classes_train = None
    classes_test = None

    for k_shots in shots:
        for dataset_type, meta in meta_dict.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            sampler = TaskSampler(meta, classes, config)

            for episode in range(num_val_episodes):
                with torch.no_grad():
                    train_data, test_data = sampler\
                        .sample_train_val_tasks(config.N_WAYS_TRAIN,
                                                config.N_WAYS_TEST,
                                                k_shots,
                                                config.N_QUERY_TRAIN, 
                                                config.N_QUERY_TEST, verbose=True)
                    data_dict = {'train': train_data, 'test': test_data}
                    for class_type, (Q_set, S_set, current_classes) in data_dict.items():
                        if class_type == 'test' and classes_test is None:
                            classes_test = current_classes
                        model.build_prototypes(S_set, model.model.backbone)
                        Q_loader = Q_set.get_dataloaders(batch_size=config.BATCH_SIZE)
                        print(current_classes)
                        evaluatorVOC = EvaluatorVOC(Q_loader, 
                                                    model, 
                                                    len(current_classes), 
                                                    current_classes)  # remove bg class

                        map_val = evaluatorVOC.eval(verbose=True)
                        metrics['{}-{}'.format(dataset_type, class_type)][k_shots].append(map_val)
    
    for k, value in metrics.items():
        if k != 'shots':
            temp_dict = {}
            for s, map_list in value.items():
                temp_dict[s] = np.array(map_list).mean()
            metrics[k] = temp_dict

    classes_train = [c for c in classes if c not in classes_test]
    print('Done in {:.2f}s.'.format(time.time() - tic))
    return classes_train, classes_test, metrics


def filter_embeddings(embeddings, labels):
    embeddings = torch.stack(embeddings)
    labels = torch.Tensor(labels).long() + 1 # for counting
    
    counts = torch.bincount(labels)
    classes = set(labels.numpy())
    max_per_class = int(counts.float().mean().item())
    ind_per_class = {c: torch.where(labels == c)[0] for c in classes}

    embeddings_per_class = [embeddings[ind][torch.randperm(ind.shape[0])][:max_per_class] for ind in ind_per_class.values()]
    labels_per_class = [c * torch.ones_like(ind)[torch.randperm(ind.shape[0])][:max_per_class] for c, ind in ind_per_class.items()]
    

    return torch.cat(embeddings_per_class, dim=0), torch.cat(labels_per_class, dim=0) - 1 