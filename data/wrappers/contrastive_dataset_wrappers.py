from torch.utils.data import Dataset
import random
from tqdm import tqdm
from ..utils import get_perturb_fn
from .base_wrappers import ShuffledDataset

class BaseContrastiveWrapper(Dataset):
    """
    Base wrapper for contrastive learning.
    """
    def __init__(self, dataset, init_shuffle=True):
        if init_shuffle:
            self.dataset = ShuffledDataset(dataset)
        else:
            self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    
class DictContrastiveWrapper(BaseContrastiveWrapper):
    """
    Base wrapper with y_dict.
    """
    def __init__(self, dataset, *args, **kwargs):
        super(DictContrastiveWrapper, self).__init__(dataset=dataset, *args, **kwargs)
        self.y_dict = self._make_y_dict()
        
    def _make_y_dict(self):
        """
        Make dictionary where keys are labels and values are indices of items with corresponding label.
        """
        y_dict = {}
        for i, (x,y) in tqdm(enumerate(self.dataset)):
            if y not in y_dict:
                y_dict[y] = []
            y_dict[y].append(i)
        return y_dict

class PairDatasetWrapper(DictContrastiveWrapper):
    """
    Dataset wrapper for pair retrieval.
    """
    def __init__(self, dataset, x1_perturb=None, x2_perturb=None, *args, **kwargs):
        super(PairDatasetWrapper, self).__init__(dataset=dataset, *args, **kwargs)
        self.x1_perturb = get_perturb_fn(x1_perturb) if x1_perturb is not None else lambda x: x
        self.x2_perturb = get_perturb_fn(x2_perturb) if x2_perturb is not None else lambda x: x
    
    def __len__(self):
        return len(self.dataset)*2

    def __getitem__(self, idx):
        """
        Returns : data[item1], label[item1], data[item2], label[item2]
        
        item1 is decided according to idx, while item2 is randomly decided.
        """
        data_idx = idx//2
        is_positive = idx%2

        x,y = self.dataset[data_idx]

        if is_positive:
            candidates = self.y_dict[y]
        else:
            candidates = []
            for key in self.y_dict:
                if key == y:
                    continue
                candidates = candidates + self.y_dict[key]
        pair_idx = random.choice(candidates)

        x2, y2 = self.dataset[pair_idx]

        return self.x1_perturb(x),y, self.x2_perturb(x2),y2
    
class TwinDatasetWrapper(BaseContrastiveWrapper):
    """
    Dataset wrapper for twin retrieval. (Same data, but maybe two different perturbations)
    Even if perturb functions are not given, two samples could be different because of the transforms function in the Dataset.
    """
    def __init__(self, dataset, x1_perturb=None, x2_perturb=None, *args, **kwargs):
        super(TwinDatasetWrapper, self).__init__(dataset=dataset, *args, **kwargs)
        self.x1_perturb = get_perturb_fn(x1_perturb) if x1_perturb is not None else lambda x: x
        self.x2_perturb = get_perturb_fn(x2_perturb) if x2_perturb is not None else lambda x: x
    def __getitem__(self, idx):
        """
        Returns : data[item1], label[item1], data[item1], label[item1]
        
        item1 is decided according to idx.
        """
        data_idx = idx

        x,y = self.dataset[data_idx]
        x2, y2 = self.dataset[data_idx]

        return self.x1_perturb(x),y, self.x2_perturb(x2),y2