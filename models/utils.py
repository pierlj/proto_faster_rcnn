import torch

def balance_classes(pos_inds, gt_labels, max_item_per_class=float('inf')):
    if max_item_per_class is not None:
        classes, counts = gt_labels[pos_inds].unique(return_counts=True)

        min_count, class_min = torch.min(counts, dim=-1)
        pos_inds_per_class = {c.item(): torch.where(gt_labels[pos_inds] == c)[0] for c in classes}

        pos_inds_balanced = []
        nb_example_per_class = min(max_item_per_class, min_count.item())
        for inds in pos_inds_per_class.values():
            shuffling = torch.randperm(inds.shape[-1])[:nb_example_per_class]
            pos_inds_balanced.append(pos_inds[inds][shuffling])
        
        pos_inds_balanced = torch.cat(pos_inds_balanced)
        shuffling = torch.randperm(pos_inds_balanced.shape[-1])
        pos_inds_balanced = pos_inds_balanced[shuffling]
        return pos_inds_balanced
    else:
        return pos_inds