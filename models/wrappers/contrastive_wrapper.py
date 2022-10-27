import torch.nn as nn
import torch
from .base_model_wrapper import ModelWrapper
from ...metrics import get_criterion, get_dist_metric
#from ...metrics.metrics import average_precision

class ContrastiveWrapper(ModelWrapper):
    """
    Model wrapper for contrastive learning
    """
    def __init__(self, criterion, embed_dim=None, dist_linear=False, dist_metric='cosine-sim', *args, **kwargs):
        super(ContrastiveWrapper, self).__init__(*args, **kwargs)
        self.criterion = get_criterion(criterion)
        self.embed_dim = embed_dim

        self.dist_linear = None
        if dist_linear:
            assert embed_dim is not None
            self.dist_linear = nn.Linear(embed_dim, 1).to(device=self.device)
        self.dist_metric = get_dist_metric(dist_metric, dist_linear=self.dist_linear)
        self.is_sim = dist_metric.split('-')[-1] == 'sim'
        
    def _model(self, x, mode):
        if mode == 'same':
            return self.model(x)
        else:
            raise NotImplementedError()
            
    def _get_eval_metric(self, metric):
        const_kwargs = {}
        metric = metric.lower()
        if metric == 'batch_ap':
            const_kwargs['dist_fn'] = self.dist_metric
            const_kwargs['is_sim'] = self.is_sim
        elif metric == 'ap':
            ## @todo ap argument feeding
            raise NotImplementedError()
        elif metric[:10] == 'ap@':
            ## @todo ap@ argument feeding
            raise NotImplementedError()
        elif metric[:16] == 'batch_ap@':
            const_kwargs['dist_fn'] = self.dist_metric
            const_kwargs['is_sim'] = self.is_sim
        else:
            return super()._get_eval_metric(metric)
        metric_fn = super()._get_eval_metric(metric)
        return lambda *args, **kwargs: metric_fn(*args, **kwargs, **const_kwargs)
    
    '''def _precision(self, xs, ys, batch=False, k=None):
        if batch:
            ## batch-wise precision (out : (1,))
            xs = torch.cat(xs,0)
            ys = torch.cat(ys,0)
            dist_ij = self.dist_metric(xs.unsqueeze(0), xs.unsqueeze(1)).squeeze(-1)
            ## dist_ij : (n_batch, n_batch)
            y_ij = ys.unsqueeze(0) == ys.unsqueeze(1)
            
            mask = torch.eye(y_ij.shape[0], dtype=torch.bool).to(device=dist_ij.device)
            dist_ij = dist_ij[~mask].view(dist_ij.shape[0],-1)
            y_ij = y_ij[~mask].view(y_ij.shape[0],-1)
            
            ## y_ij : (n_batch, n_batch)
            ap = average_precision(dist_ij[y_ij], dist_ij[~y_ij], k=k, largest=self.is_sim)
            return ap
        else:
            ## TODO : row-wise precision (out : (n_batch,))
            xs = torch.stack(xs, 1)
            ys = torch.stack(ys, 1)
            raise NotImplementedError()'''