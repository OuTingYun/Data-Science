# Crowd Estimation


## File

```
- 311551087
- MyDataset(雲端硬碟下載)
        - train
        - test
        - val
    - 311551087.sh
    - model.pth
    - requirements.txt
    - test.py
    - models.py
    - crow.py
    - readme.pdf
- 311551087.csv (submiot file) - train(雲端硬碟下載)

```
## Testing

Run the file `run.sh` then you will see the following content in the terminal

```
number of img: 1772
```

then the result will store in `result.csv`

## Training

1. Download the train.zip and unzip the file
2. Into the directory of `train`
3. run `train.sh`
4. the model will store in `./ckps/input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-120_normCood-0`

