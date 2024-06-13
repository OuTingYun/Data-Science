import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter
import pandas as pd
import numpy 
import random
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torchmetrics.classification import BinaryPrecision
from  torch.optim.lr_scheduler import ExponentialLR,StepLR
import os
from torch_geometric.nn import GATConv
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,cached=True,)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(10,13,2).to(device)
model = torch.load('model_last_0.815.pt')
# print("finish")
tesk_mask = numpy.load('dataset/test_mask.npy')
test_data = torch.load('dataset/test_sub-graph_tensor_noLabel.pt')


test_pred = model(test_data.feature.float().to(device),test_data.edge_index.to(device)).to('cpu')
m = torch.nn.Softmax(dim=1)
output = m(test_pred)

df = pd.DataFrame(output[:,-1].detach().numpy(),columns=['node anomaly score'])
df[tesk_mask].to_csv(f"submit.csv",index=True,index_label='node idx') #save to file
