import torch
from torch.utils.data import  Dataset,DataLoader
from torchvision import transforms
import pandas as pd
from resnet12 import ResNet12
import argparse
import pickle

class cfg():
    def __init__(self):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
     


class testDataset(Dataset):
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
    
def getData(root):
   
    test_set = testDataset(
        root= root+'/test.pkl',
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    return test_loader



def predict():
    # We'll count everything and compute the ratio at the end

    
    print("predicting...")
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
    df.to_csv('submit.csv',index=False)

        
CFG = cfg()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data', help='dataset root')
    args = parser.parse_args()

    test_loader = getData(args.data)    
    model = ResNet12([64, 160, 320, 640]).to(CFG.device)
    model.load_state_dict(torch.load('./models/ep1400_078_124.pth'))
    predict()
    
