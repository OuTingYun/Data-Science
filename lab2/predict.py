import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F
import pandas as pd
import torch.nn.utils.prune as prune
import datetime
import os
import sys

class cfg():
    def __init__(self):
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transformer = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]
        )
        self.testloader = self.getData()

        self.optim = "NAdam"
        self.epoch = 30
        self.prune_ratio = 0.58

    def getData(self):
     
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=self.transformer)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=0)
        return testloader

class utils():
    def evaluation(self,net,submit=False,name=""):
        net.eval()
        correct = 0
        total = 0
        pred_arr = []
        with torch.no_grad():
            for data in CFG.testloader:
                images, labels = data
                images, labels = images.to(CFG.device), labels.to(CFG.device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pred_arr.append(predicted.item())
        accuracy = 100 * correct / total
        if submit:
            pred_data = {"pred":pred_arr}
            df_pred = pd.DataFrame(pred_data)
            df_pred.to_csv(f'{name}.csv', index_label='id')
        return accuracy
    def countZeroWeights(self,model):
        zeros = 0
        for param in model.parameters():
            if param is not None:
                zeros += torch.sum((param == 0).int())
        pytorch_total_params = sum(p.numel() for p in model.parameters()) 
        return pytorch_total_params-zeros


class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.resnet50 = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, 10)

        def forward(self, x):
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)

            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)

            x = self.resnet50.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet50.fc(x)
            return x
           
class ResNet_S(nn.Module):
    def __init__(self):
        super(ResNet_S, self).__init__()
        self.conv1= teacher.resnet50.conv1
        self.bn1 = teacher.resnet50.bn1
        self.relu= teacher.resnet50.relu
        self.maxpool= teacher.resnet50.maxpool

        self.layer1 = teacher.resnet50.layer1[0:1]

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.avgpool= teacher.resnet50.avgpool
        self.fc= nn.Linear(512,10,bias=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)      
        return x
    
if __name__ == "__main__":

    path = 'model.pth'
    name = 'submit'

    CFG = cfg()
    UTILS = utils()


    teacher = ResNet().to(CFG.device)
    checkpoint = torch.load("./resnet-50.pth")
    teacher.load_state_dict(checkpoint['model_state_dict'])

    st = ResNet_S().to(CFG.device)
    checkpoint = torch.load(path)
    st.load_state_dict(checkpoint['model_state_dict'])

    print(f"params:{UTILS.countZeroWeights(st)}")
    print("Test acc:",UTILS.evaluation(st,submit=True,name=name))

