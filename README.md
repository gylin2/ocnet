# OCNet-DSP
This repository contains the official implementation of our paper, "One-Class Network with Directed Statistics Pooling for Spoofing Speech Detection".

## Run the training procedure
```bash
python3 train.py -p path_to_your_database -o ./models/output  --gpu 0 --warmup
```
## Run the test procedure
```bash
python3 test.py -p path_to_your_database -m ./models/output --gpu 0 --part eval
```
## Run the cross-dataset procedure
```bash
python3 cross_dataset_2021.py -p path_to_your_database -m ./models/output -a LA --gpu 0 --phase eval
```

```bash
python3 cross_dataset_2015.py -p path_to_your_database -m ./models/output --gpu 0 --part eval
```


## Citation
If you find this work useful in your research, please consider citing this paper.

## Acknowledgement
This code refers to the following two projects:

[1] https://github.com/yzyouzhang/AIR-ASVspoof

[2] https://github.com/clovaai/aasist
