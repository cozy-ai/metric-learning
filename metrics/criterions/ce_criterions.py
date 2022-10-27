import torch.nn as nn
import torch
import torch.nn.functional as F

class PairwiseCE(nn.Module):
    """
    Pairwise crossentropy loss (with softmax)
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, dist, y1, y2):
        """
        Compute crossentropy between given distance against label (1 if different 0 if same)
        """
        y = (y1!=y2).to(device=dist.device, dtype=torch.float)
        return self.bce(dist, y)

class InfoNCE(nn.Module):
    """
    Noise Contrastive Estimator loss
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, dist, y1, y2):
        if len(dist.size()) == 1:
            ## dist : (n_batch, )
            ## y: (n_batch, )
            raise NotImplementedError()
        elif len(dist.size()) == 2:
            y = (y1==y2).to(device=dist.device, dtype=torch.bool)
            ## dist : (N, N)
            ## y : (N, N)
            
            ## mask out diagonal elements
            ## @todo (?): genearlize to cases where y1 and y2 are from different samples
            mask = torch.eye(y.shape[0], dtype=torch.bool).to(device=dist.device)
            dist = dist[~mask].view(dist.shape[0],-1)
            y = y[~mask].view(y.shape[0],-1)
            ## dist : (N, N-1)
            ## y : (N, N-1)
            
            ## true score for each row i : where (y==1)
            y = (y==1).nonzero()[:,1]
            #print(dist.shape, y.shape)
            return self.ce(dist/0.07, y)