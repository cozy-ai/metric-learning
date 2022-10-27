from torch.utils.data import Dataset
import numpy as np

class ShuffledDataset(Dataset):
    """
    Retrieve items in shuffled order.
    The order is not shuffled after init.
    """
    def __init__(self, dataset):
        self.dataset=dataset
        self.shuffled_idcs = np.arange(len(dataset))
        np.random.shuffle(self.shuffled_idcs)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[self.shuffled_idcs[idx]]