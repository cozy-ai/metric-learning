import torch

def average_precision(pred_pos, pred_neg, k=None, largest=True):
    """
    Return Average Precision from given predictions (positive/negative).
    """
    ## pred_pos, pred_neg : (n,) => ns can be different
    
    ## sort or topk depending on k 
    ## accumulate true label from top til k -> k-th val : total tp when cutoff is k (tp+fp = k)
    ## divide with arange tensor => tp/(tp+fp)
    ## mask with true label and sum
    ## divide with norm=min(n_gt, k)
    
    if k is None or pred_pos.shape[0]+pred_neg.shape[0] <= k:
        sorted_idx = torch.cat([pred_neg, pred_pos]).sort(descending=largest)[1]
    else:
        sorted_idx = torch.cat([pred_neg, pred_pos]).topk(k, sorted=True, largest=largest)[1]
    gt_mask = (sorted_idx>=pred_neg.shape[0]).to(dtype=torch.float)
    precision = torch.cumsum(gt_mask, 0) / (torch.arange(gt_mask.shape[0])+1).to(device=gt_mask.device)
    
    norm = min(pred_pos.shape[0], k) if k is not None else pred_pos.shape[0]
    
    ap = (precision*gt_mask).sum() / norm
    return ap.item()

## @todo generalize ap functions
## @todo organize inputs/outputs of ap functions
def ap(xs, ys, k=None):
    ## @todo row-wise average precision (out : (n_batch,))
    xs = torch.stack(xs, 1)
    ys = torch.stack(ys, 1)
    raise NotImplementedError()

def batch_ap(xs, ys, dist_fn, is_sim, k=None):
    ## batch-wise average precision (out : (1,))
    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)
    dist_ij = dist_fn(xs.unsqueeze(0), xs.unsqueeze(1)).squeeze(-1)
    ## dist_ij : (n_batch, n_batch)
    y_ij = ys.unsqueeze(0) == ys.unsqueeze(1)
    
    mask = torch.eye(y_ij.shape[0], dtype=torch.bool)
    dist_ij = dist_ij[~mask].view(dist_ij.shape[0],-1)
    y_ij = y_ij[~mask].view(y_ij.shape[0], -1)
    
    ## y_ij : (n_batch, n_batch)
    ap = average_precision(dist_ij[y_ij], dist_ij[~y_ij], k=k, largest=is_sim)
    return ap