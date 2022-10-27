from tqdm import tqdm
import os
import time
import torch
from ...metrics import get_eval_metric

## @todo resume training

class ModelWrapper:
    """
    Basic Model Wrapper
    """
    def __init__(self, model, e=0, device='cpu', log_path='./train_logs', model_name=None, eval_metrics=[], criteria_score=None, criteria_loss=None, **kwargs):
        """
        Build wrapper instance and make log directories.
        """
        self.model = model.to(device=device)
        self.e=e
        self.device=device
        self.log_path=log_path
        self.eval_metrics = eval_metrics
        
        assert criteria_score is None or criteria_loss is None, "only one must be given criteria_score or criteria_loss"
        ## criteria_metric : the higher the better
        if criteria_loss is not None:
            self.criteria_metric = criteria_loss
            self.criteria_sign = -1
        else:
            self.criteria_metric = criteria_score
            self.criteria_sign = 1
        self.best_criteria = float('-inf')
        
        for key in kwargs:
            setattr(self, key, kwargs[key])

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if model_name is None or os.path.exists(f'{log_path}/{model_name}'):
            model_idx = 0
            base_name = 'mymodel' if model_name is None else model_name
            while True:
                model_name = f'{base_name}_{model_idx}'
                if os.path.exists(f'{log_path}/{model_name}'):
                    model_idx += 1
                    continue
                else:
                    break

        self.model_name = model_name
        os.makedirs(f'{log_path}/{model_name}')
        os.makedirs(f'{log_path}/{model_name}/weights')
        os.makedirs(f'{log_path}/{model_name}/optims')
        
    def run_train(self, train_loader, optimizer, epoch, val_loader=None, scheduler=None):
        """
        Run model training.
        """
        model = self.model
        for self.e in range(self.e, epoch):
            ts=time.time()
            model.train()
            train_metrics = self.epoch_step(train_loader, optimizer)
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    val_metrics = self.epoch_step(val_loader)
            
            metrics = [('train', train_metrics)]
            if val_loader is not None:
                metrics.append(('val', val_metrics))
            
            if scheduler is not None and self.e >= scheduler.warm_up:
                scheduler.step()
            
            is_best=False
            if self.criteria_metric is not None and val_loader is not None:
                criteria = self.criteria_sign * val_metrics[self.criteria_metric]
                if criteria>self.best_criteria:
                    self.best_criteria = criteria
                    is_best = True
            
            ts = time.time()-ts
            self.log_epoch(metrics=metrics, timestamp=ts, optimizer=optimizer, is_best=is_best)

    def _get_batchsize(self, loaded_data):
        return loaded_data[0].shape[0]

    def epoch_step(self, dataloader, optimizer=None):
        """
        Unit operation for dataloader in one epoch.
        """
        n_samples = 0.
        avg_loss = 0.
        avg_metrics = {}
        
        pbar = tqdm(dataloader)
        
        for i, loaded_data in enumerate(pbar):
            ts = time.time()
            loss, metrics = self.batch_step(loaded_data, optimizer=optimizer)
            batchsize = self._get_batchsize(loaded_data)
            avg_loss = (avg_loss*n_samples + loss*batchsize) / (n_samples+batchsize)
            for metric in metrics:
                if metric not in avg_metrics:
                    avg_metrics[metric] = 0.
                    
                avg_metrics[metric] = (avg_metrics[metric]*n_samples + metrics[metric]*batchsize) / (n_samples + batchsize)
            n_samples += batchsize
            description = 'l=%.4g'%avg_loss
            for metric in avg_metrics:
                description = description+',  %s=%0.4g'%(metric, avg_metrics[metric])
            pbar.set_description(description)
        avg_metrics['Loss'] = avg_loss
        return avg_metrics
    
    def batch_step(self, loaded_data, optimizer=None):
        """
        Unit operation for loaded data in one batch.
        Must be overrided.
        """
        raise NotImplementedError()
        
    def _eval_metric(self, metric, *args, **kwargs):
        metric_fn = self._get_eval_metric(metric)
        return metric_fn(*args, **kwargs)
    
    def _get_eval_metric(self, metric):
        return get_eval_metric(metric)
    
    def log_epoch(self, metrics=None, timestamp=None, optimizer=None, is_best=False):
        """
        Log training progress on both console output and filesystem.
        """
        ## @todo save training states such as self.e, self.best_criteria, etc.
        
        if timestamp is not None:
            print('(%.2fs) '%timestamp, end='')
        print('[Epoch %d]'%(self.e+1))
        for metrics_i in metrics:
            if metrics is not None:
                split, split_metrics = metrics_i
                print('\t(%s) '%split, end='')
                for metric in split_metrics:
                    print('\t%s : %.4g'%(metric, split_metrics[metric]), end='')
                print('')
        
        #self.save_model_state(f'{self.log_path}/{self.model_name}/weights/model_e{self.e+1}.pth')
        self.save_model_state(f'{self.log_path}/{self.model_name}/weights/latest.pth')
        if is_best:
            self.save_model_state(f'{self.log_path}/{self.model_name}/weights/best.pth')
        self.log_progress(metrics, f'{self.log_path}/{self.model_name}/progress.log')
        if optimizer is not None:
            #self.save_optim_state(optimizer.state_dict(), f'{self.log_path}/{self.model_name}/optims/optimizer_e{self.e+1}.pth')
            self.save_optim_state(optimizer.state_dict(), f'{self.log_path}/{self.model_name}/optims/optimizer_latest.pth')
    
    def save_model_state(self, path):
        return torch.save(self.model.state_dict(), path)
        
    def log_progress(self, metrics, path):
        mode = 'a' if os.path.isfile(path) else 'w'
        with open(path, mode) as f:
            f.write('[Epoch %d]\n'%(self.e+1))
            for split, split_metrics in metrics:
                f.write('\t(%s) '%split)
                for metric in split_metrics:
                    f.write('\t%s : %.4g'%(metric, split_metrics[metric]))
                f.write('\n')

    def save_optim_state(self, optim_state, path):
        return torch.save(optim_state, path)
    