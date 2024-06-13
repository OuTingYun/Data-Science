# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import datetime

import json

from rich.console import Console
import evaluate
console = Console(record=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class cfg():
    def __init__(self):
        self.model_params = {
            # "MODEL": "google/t5-v1_1-base",             # model_type: t5-base/t5-large
            "MODEL":"t5-base",
            "TRAIN_BATCH_SIZE": 7,          # training batch size
            "VALID_BATCH_SIZE": 7,          # validation batch size
            "TRAIN_EPOCHS": 10,              # number of training epochs
            "VAL_EPOCHS": 1,                # number of validation epochs
            "LR": 1e-4,          # learning rate
            "MAX_SOURCE_TEXT_LENGTH": 512,  
            "MAX_TARGET_TEXT_LENGTH": 50,
            "SEED": 42,                     # set seed for reproducibility
        }
      


# run = Run()
CFG = cfg()


class DataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and 
    loading it into the dataloader to pass it to the neural network for finetuning the model

    """

    def __init__(self, fdir, tokenizer, source_len, target_len):
        self.data = []
        
        with open(fdir, 'r') as reader:
            for _,row in enumerate(reader):
                try:
                    self.data.append(json.loads(row))
                except:
                    print("empty?",_)
                    self.data.append("")
     
        self.num = len(self.data)
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        data = self.data[index]
        try:
            source_text = str(data['body'])
            target_text = ""
        except:
            print("problem:",index,"---",data['body'])
            source_text = ""
            target_text = ""

        # cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }






def getDataset(tokenizer):
    model_params = CFG.model_params
    test_set = DataSetClass('test.json', tokenizer,
                           model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"])

    test_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
    }


    test_loader = DataLoader(test_set, **test_params)
    return test_loader
def test(fname,tr,tokenizer, model):
    model.eval()
    predictions = []
    actuals = []
    console.print(f'Testing..', end='\r')
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):

            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

            predictions.extend(preds)
            if _ %20==0:
                console.print(f'test iter..{_}', end='\r')

    data = [{"title":pred } for pred in predictions]
    
    with open(f"{fname}.json", "w",encoding='utf8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
        


if __name__ == '__main__':

    tokenizer = T5Tokenizer.from_pretrained("./model_ep9")
    test_loader = getDataset(tokenizer)
    model = T5ForConditionalGeneration.from_pretrained("./model_ep9")
    model = model.to(device)
    model.eval()
    test(f"311551087-test",tr="", tokenizer=tokenizer, model=model)





