import torch
from .contrastive_wrapper import ContrastiveWrapper

class PairContrastiveWrapper(ContrastiveWrapper):
    """
    Model wrapper for contrastive learning that uses pair inputs.
    """
    def __init__(self, second_model='same', complete_loss=None, *args, **kwargs):
        super(PairContrastiveWrapper, self).__init__(*args, **kwargs)
        self.second_model = second_model.lower()
        assert self.second_model in ['same', 'mean_teacher', 'freeze']
        self.complete_loss = complete_loss
        if complete_loss is not None:
            assert complete_loss in ['label', 'sample']
        
    def batch_step(self, loaded_data, optimizer=None):
        model = self.model

        x1,y1, x2,y2 = loaded_data

        x1 = x1.to(device=self.device, dtype=torch.float)
        x2 = x2.to(device=self.device, dtype=torch.float)

        out1 = model(x1)
        out2 = self._model(x2, self.second_model)
        
        xs = [out1, out2]
        ys = [y1, y2]
        
        if self.complete_loss is not None:
            out1 = torch.cat([out1, out2], dim=0)
            out2 = out1.unsqueeze(1)
            out1 = out1.unsqueeze(0)
            if self.complete_loss == 'label':
                y1 = torch.cat([y1, y2], dim=0)
            elif self.complete_loss == 'sample':
                y1 = torch.cat([torch.arange(y1.shape[0]) for _ in range(2)], dim=0)
            y2 = y1.unsqueeze(1)
            y1 = y1.unsqueeze(0)
        dist = self.dist_metric(out1, out2).squeeze(-1)
        ## dist shape : (n_batch, ) or (n_batch*2, n_batch*2) => row-wise / complete pair-wise

        ## @todo generalize criterion
        loss = self.criterion(dist, y1, y2)

        metrics = {}
        for metric in self.eval_metrics:
            metrics[metric] = self._eval_metric(metric, xs=xs, ys=ys)
            
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), metrics