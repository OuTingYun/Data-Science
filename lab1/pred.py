from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms

from PIL import Image
import sys

class DataSet(Dataset):
    def __init__(self,imgs_path,transform=None):
        self.imgs_path = imgs_path
        self.transform = transform
    def __len__(self):
        return len(self.imgs_path)
    def __getitem__(self,idx):
       
        path = self.imgs_path[idx]
        img = Image.open(path, mode='r').convert("RGB")
        if transforms!=None:
            img = self.transform(img)
        return img

def get_data(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [item.replace("\n","") for item in lines]
    lines = [item for item in lines if item!="" ]

    
    test_dataset = DataSet(lines,CFG.transforms_test)
    test_loader = DataLoader(test_dataset,batch_size=1)
    return test_loader


class cfg():
    def __init__(self):
        self.transforms_train = transforms.Compose([
            transforms.Resize((512, 512)),   #must same as here
            # transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
        ])
        self.transforms_test = transforms.Compose([
            transforms.Resize((512, 512)),   #must same as here
            # transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
def get_model():
    model = torch.load("resnet18(last_f1).pt",map_location=torch.device('cpu'))
    return model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def predict(model,test_loader):
    model = model.to(device)
    preds = ""
    for img in test_loader:
        img = img.to(device)
        pred = model(img).squeeze()
        threshold = torch.tensor(0.5)
        results = (pred>threshold).float()*1

        preds+=str(int(results.item()))
    print(preds)
    with open("311551087.txt","w") as f:
        f.write(preds) 
CFG = cfg()
if __name__ == '__main__':
    args = sys.argv
    model = get_model()
    test_loader = get_data(args[1])
    predict(model,test_loader)

        

