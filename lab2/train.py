import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
from torchinfo import summary
from tqdm import tqdm
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
        self.folder_name = self.getFolder()
        self.trainloader,self.testloader = self.getData()

        self.optim = "Adam"
        self.epoch = 50
        self.prune_ratio = 0.58
    def getData(self):
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=self.transformer)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                    shuffle=True, num_workers=8)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=self.transformer)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=0)
        return trainloader,testloader
    
    def getFolder(self):
        now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
        folder_name = 'models/'+now.strftime('%m%d-%H%M')
        return folder_name

class utils():
    def evaluation(self,net,submit=False):
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
            df_pred.to_csv('train_pred.csv', index_label='id')
        return accuracy
    def countZeroWeights(self,model):
        zeros = 0
        for param in model.parameters():
            if param is not None:
                zeros += torch.sum((param == 0).int())
        pytorch_total_params = sum(p.numel() for p in model.parameters()) 
        return pytorch_total_params-zeros
    def calc_param(self,student_model):
        total_count = sum([p.numel() for p in student_model.parameters()])
        prune_count = 0
        for _, module in student_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune_count += torch.sum(module.weight == 0)
        return total_count - prune_count

    def ACC(self,pred,y):
        _, _pred = torch.max(pred, 1)
        correct= (_pred == y).sum().item()
        return correct     
    def pruning(self,last=False):
        parameters_to_prune = (
                (student.conv1, 'weight'),
                (student.layer1[0].conv1,'weight'),
                (student.layer1[0].conv2,'weight'),
                (student.layer1[0].conv3,'weight'),
                (student.layer1[0].downsample[0],'weight'),
                (student.layer3[0], 'weight'),
                (student.fc, 'weight')
            )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=CFG.prune_ratio,
        )
        if last:
            summary(student, input_size=(128,3, 28, 28))
        for item in parameters_to_prune:
            prune.remove(item[0],item[1])
    

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

def train():

    teacher = ResNet().to(CFG.device)
    checkpoint = torch.load("./resnet-50.pth")
    teacher.load_state_dict(checkpoint['model_state_dict'])
    
    UTILS.pruning()
    print(UTILS.countZeroWeights(student))

    if CFG.optim == "Adam":
        # setting optimizer
        optimizer_s = optim.Adam(student.parameters())

    # setting scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s, factor=0.5, patience=2, verbose=True, min_lr= 0.00001
    )
    # add Softmax
    sft = nn.Softmax(dim=1)
    
    correct_s, correct_t = 0, 0
    total = 0
    best = 91
    

    for epoch in range(CFG.epoch):

            print("EP:",epoch,"...",end=" ")
            teacher.eval()
            student.train()
            loss_sum = 0
            for idx,(x,y) in enumerate(CFG.trainloader):
                    
                    x=x.to(CFG.device)
                    y=y.to(CFG.device)

                    pred_t = teacher(x)
                    
                    pred_ts = sft(pred_t)

                    pred_s = student(x)
                    pred_ss = sft(pred_s)

                    # loss_ht = F.cross_entropy(pred_ts,y)
                    loss_hs = F.cross_entropy(pred_ss,y)
                    loss_kd = F.kl_div(pred_ss,pred_ts)

                    loss = loss_hs+loss_kd
                    loss_sum += loss.item()

                    optimizer_s.zero_grad()
                    loss.backward()
                    optimizer_s.step()

                   
                    total += y.size(0)
                    correct_s += UTILS.ACC(pred_ss,y)
                    correct_t += UTILS.ACC(pred_ts,y)


            if epoch == CFG.epoch-1:
                UTILS.pruning(last=True)      
            else:
                UTILS.pruning()      

            accuracy_s = 100 * correct_s / total
            accuracy_t = 100 * correct_t / total
            accuracy_s_test = UTILS.evaluation(student)
            
            scheduler.step(loss_sum)
            
            print({
                'loss':f'{loss_sum:.2f}',
                'Sacc Train/Test':f'{accuracy_s:.2f}%/{ accuracy_s_test:.2f}%',
                'Tacc Train':f'{accuracy_t:.2f}'})
  
            if best<accuracy_s_test:
                best = accuracy_s_test
        

if __name__ == "__main__":

    CFG = cfg()
    UTILS = utils()
    

    teacher = ResNet().to(CFG.device)
    checkpoint = torch.load("./resnet-50.pth")
    teacher.load_state_dict(checkpoint['model_state_dict'])

    student = ResNet_S().to(CFG.device)
    # student.load_state_dict(torch.load("models/0314-2117/model91.70.pth")['model_state_dict'])
    

    train()
    
    print("params:",UTILS.countZeroWeights(student))
    print("Test acc:",UTILS.evaluation(student,submit=True))

