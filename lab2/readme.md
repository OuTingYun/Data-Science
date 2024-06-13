## Model Compression 


### Folder

put pretrined `resnet-50.pth` and `model.pth` in the same directory
```
- hw2
    - model.pth
    - predict.py
    - train.py
    - resnet-50.pth
```

### Training

Run the following instruction

```bash
python train.py
```

We will not save any model, only show the result on terminal

### Testing

Load the model `model.pth` and evaluate the accuracy on testing dataset

Run the following instruction

```
python predict.py
```

### Model Size

After compress the model, we have 94,560 params which less then 100,000.

