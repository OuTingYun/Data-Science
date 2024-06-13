import torch
from torch import nn, optim,Tensor

from torch.utils.data import  DataLoader, Sampler, Dataset
from torchvision import transforms

from tqdm import tqdm

from dataset import trainDataset, testDataset

import numpy as np

from typing import List
import pandas as pd

import datetime
import os
from resnet12 import ResNet12


import random
from typing import List, Tuple, Iterator

from abc import abstractmethod
from typing import List, Tuple

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
    



class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: FewShotDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            ).tolist()

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )



class cfg():
    def __init__(self):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
 
        self.n_way = 5  # Number of classes in a task
        self.n_shot = 5  # Number of images per class in the support set
        self.n_query = 10  # Number of images per class in the query set
        self.n_train_tasks = 8000
        self.n_valid_tasks = 100

        # hyper-parameters
        self.lr = 0.001
        self.supervise_lr = 0.001
        self.pretrain_bs = 128
        self.pretrain_epoch = 25
        self.preval_epoch = 5
        self.method = 'cdist-way'
        self.criterion = "CrossEntropyLoss"
        self.optimizer = "Adam-resnet12"

        now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
        self.folder_name = 'models/'+now.strftime('%m%d-%H%M')
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        else:
            assert("Making file existing")
    

def getData():
    train_set = trainDataset(
        root='./data/train.pkl',
        transform=transforms.Compose([transforms.ToTensor()])
    )
    val_set = trainDataset(
        root='./data/validation.pkl',
        transform=transforms.Compose([transforms.ToTensor()])
    )
    test_set = testDataset(
        root='./data/test.pkl',
        transform=transforms.Compose([transforms.ToTensor()])
    )

    train_sampler = TaskSampler(train_set, n_way=CFG.n_way, n_shot=CFG.n_shot,
                                n_query=CFG.n_query, n_tasks=CFG.n_train_tasks)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    pretrain_loader = DataLoader(
        train_set,
        shuffle=True,
        num_workers = 12,
        pin_memory = True,
        batch_size=CFG.pretrain_bs,
    )


    val_sampler = TaskSampler(val_set, n_way=CFG.n_way, n_shot=CFG.n_shot,
                              n_query=CFG.n_query, n_tasks=CFG.n_valid_tasks)
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    return pretrain_loader,train_loader, val_loader, test_loader


def sliding_average(value_list: List[float], window: int) -> float:

    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def predict(acc=None,epoch=None,mode=None):
    # We'll count everything and compute the ratio at the end

    model.eval()
    total_list = []
    task = 0
    with torch.no_grad():
        for (support_images,support_labels,true_class_ids,query_images) in test_loader:
            task+=1
            support_images = support_images.squeeze(0).to(CFG.device)
            support_labels = support_labels.squeeze(0).to(CFG.device)
            query_images = query_images.squeeze(0).to(CFG.device)

            pred_index = torch.max(model(support_images.cuda(), support_labels.cuda(
            ), query_images.cuda()).detach().data, 1)[1]
            total_list += [true_class_ids[idx].cpu().item() for idx in pred_index]
    df = pd.DataFrame({"Id":[i for i in range(len(total_list))],"Category":total_list})
    if acc!=None:
        df.to_csv(CFG.folder_name+f'/submit{epoch}_{acc}_{mode}.csv',index=False)
    else:
        df.to_csv(CFG.folder_name+'/total_best_model_submit.csv',index=False)

def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    model.eval()
    total_list = []
    with torch.no_grad():
        for episode_index, data in enumerate(data_loader):

            (support_images, support_labels,
             query_images, query_labels, class_ids) = data

            total_list.append(torch.unique(support_labels))
            total = len(query_labels)
            # print("evaluation",type(model.model))
            correct = (
                torch.max(model(support_images.cuda(), support_labels.cuda(
                ), query_images.cuda()).detach().data, 1)[1]
                == query_labels.cuda()).sum().item()

            total_predictions += total
            correct_predictions += correct

    acc = correct_predictions/total_predictions

    return round(acc, 2)


def pretrain():
    # Train the model yourself with this cell
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.supervise_lr)
    print("pretrain")
    val_best = 0
    save_model = None
    
    for epoch in range(CFG.pretrain_epoch):
        pretrain_acc = 0
        pretrain_loss = 0
        count = 0
        with tqdm(total=len(pretrain_loader)) as train_enum:
            model.train()
            for batch, (img,label) in enumerate(pretrain_loader):
                img = img.to(CFG.device)
                label = label.to(CFG.device)
                count+=len(label)

                optimizer.zero_grad()
                pred = model.pretrain(img)
                
                loss = criterion(pred,label)

                loss.backward()
                optimizer.step()

                acc = (torch.max(pred,dim=1).indices==label).sum().item()
                pretrain_acc  += acc
                pretrain_loss += loss.item()

                train_enum.set_description(f"EP {epoch}")
                train_enum.update()
            
            
  

            model.eval()
            valid_acc = evaluate(valid_loader)
            if valid_acc>val_best:
                val_best = valid_acc
                save_model = model
            
            train_enum.set_postfix({'loss':f'{pretrain_loss:.2f}','pt-acc':f'{pretrain_acc/count:.2f}',"v_acc":valid_acc})
    torch.save(save_model.state_dict(), CFG.folder_name+f'/pretrain_bestmodel_{val_best}.pth')
    return save_model
        
def train(fz=""):
    # Train the model yourself with this cell
    all_loss = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

    best = 0.
    save_model = None
    p = 0
    nnp = 0
    for param in model.parameters():
      if param.requires_grad:
        p+= np.prod(param.size())
      else:
        nnp+= np.prod(param.size())
    print("train-param:",p)
    print("non-train param:",nnp)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.8, patience=3, verbose=True)

    for episode_index, data in enumerate(train_loader):
        model.train()
        (support_images, support_labels, query_images, query_labels, _) = data
        optimizer.zero_grad()
        # print(type(model.model))
        classification_scores = model(
            support_images.cuda(), support_labels.cuda(), query_images.cuda()
        )

        loss = criterion(classification_scores, query_labels.cuda())
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())

        if episode_index % 100 == 0 or episode_index==CFG.n_train_tasks-1:
            model.eval()
            
            valid_acc = evaluate(valid_loader)
            if valid_acc > best:
                best = valid_acc
                save_model = model

        if episode_index % 100 == 0 or episode_index==CFG.n_train_tasks-1:
            ls = sliding_average(all_loss, 100)
            # tqdm_train.set_postfix(loss=ls,t_acc = train_acc,v_acc=valid_acc)

            print({
                "task:": episode_index,
                "t_loss:": round(ls,2),
                "v_acc:": valid_acc,
            })


            # scheduler.step(ls)
  
    torch.save(save_model.state_dict(), CFG.folder_name+f'/train_bestmodel_{best}.pth')
    return save_model



CFG = cfg()
if __name__ == '__main__':


    pretrain_loader, train_loader, valid_loader, test_loader = getData()
    
    model = ResNet12([64, 160, 320, 640]).to(CFG.device)
    pretrain()
    # model.load_state_dict(torch.load('models/0411-1115/pretrain_bestmodel_0.71.pth'))

    print("freeze:1,2,3")

    freeze_layer = [model.layer1, model.layer2, model.layer3]
    unfreeze_layer = [ model.layer4]

    for layer in freeze_layer:
        for param in layer.parameters():
            param.requires_grad = False
            
    for layer in unfreeze_layer:
        for param in layer.parameters():
            param.requires_grad = True
    model = train("123")
    predict()
    
