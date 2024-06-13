import argparse
import torch
import os
import numpy as np
import crowd as crowd
from models import vgg19
import pandas as pd

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--crop-size', type=int, default=512,
                    help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='./model.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str,
                    default='./MyDataset',
                    help='saved model path')
parser.add_argument('--pred-density-map-path', type=str, default='',
                    help='save predicted density maps when pred-density-map-path is not empty.')

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = args.model_path
crop_size = args.crop_size
data_path = args.data_path

dataset = crowd.testCrowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
 
dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                         num_workers=1, pin_memory=True)

model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
ans = []
id = []
id_count = 0
for inputs,  name in dataloader:
    inputs = inputs.to(device)
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        outputs, _ = model(inputs)
        
        
        ans.append(torch.sum(outputs).item())
        id_count+=1
        id.append(id_count)
    print(f"{id_count}==>{len(dataloader)}",end='\r')
pd.DataFrame({'ID': id, 'Count': ans}).to_csv(f'311551087.csv',index=False)