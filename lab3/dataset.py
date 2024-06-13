from abc import abstractmethod
from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset
import torch
import pickle


class FewShotDataset(Dataset):
    """
    Abstract class for all datasets used in a context of Few-Shot Learning.
    The tools we use in few-shot learning, especially TaskSampler, expect an
    implementation of FewShotDataset.
    Compared to PyTorch's Dataset, FewShotDataset forces a method get_labels.
    This exposes the list of all items labels and therefore allows to sample
    items depending on their label.
    """

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __getitem__ method."
        )

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __len__ method."
        )

    @abstractmethod
    def get_labels(self) -> List[int]:
        raise NotImplementedError(
            "Implementations of FewShotDataset need a get_labels method."
        )
    
class trainDataset(FewShotDataset):
    def __init__(self,root,transform):
        with open(root, "rb") as f:
            data = pickle.load(f) # a dictionary
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image = torch.tensor(self.images[idx])
        label = self.labels[idx]
        return image,label
    def get_labels(self):
        return [self.labels[idx] for idx in range(len(self.images))]
        
class testDataset(FewShotDataset):
    def __init__(self,root,transform):
        with open(root, "rb") as f:
            data = pickle.load(f) # a dictionary
        self.sup_images = data['sup_images']
        self.sup_labels = data['sup_labels']
        self.qry_images = data['qry_images']
        self.transform = transform
    def __len__(self):
        return len(self.sup_images)
    def __getitem__(self,idx):
        sup_image = torch.tensor(self.sup_images[idx])
        sup_label = self.sup_labels[idx]

        true_class_ids = list({x for x in sup_label})
        ids_labels =  torch.tensor(
            [true_class_ids.index(x) for x in sup_label]
        )
        qry_image = torch.tensor(self.qry_images[idx])

        return sup_image,ids_labels,true_class_ids,qry_image