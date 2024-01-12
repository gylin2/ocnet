# OCNet-DSP
This repository contains the official implementation of our paper, "One-Class Network with Directed Statistics Pooling for Spoofing Speech Detection". [[Paper link](https://ieeexplore.ieee.org/document/10387499)]

## Requirements
Installing dependencies:
```
pip install -r requirements.txt
```

## Run the training procedure
Run the `train.py` to train the model on the ASVspoof2019 LA dataset:
```
python3 train.py -p path_to_your_database -o ./models/new_path  --gpu 0 --warmup
```
## Run the test procedure
Run the `test.py` to evaluate the pre-trained model on the ASVspoof2019 LA dataset:
```
python3 test.py -p path_to_your_database -m ./models/OCNet --gpu 0 --part eval
```
## Run the cross-dataset procedure
To evaluate the cross-dataset performance on the ASVspoof2021 LA dataset:
```
python3 cross_dataset_2021.py -p path_to_your_database -m ./models/OCNet -a LA --gpu 0 --phase eval
```

To evaluate the cross-dataset performance on the ASVspoof2015 dataset:
```
python3 cross_dataset_2015.py -p path_to_your_database -m ./models/OCNet --gpu 0 --part eval
```

## Print the model parameters
You can run the 'model_param.py' to print the structure and parameters of the given model:
```
pyhton3 model_param.py --gpu 0
```

## Citation
If you find this work useful in your research, please consider citing this paper.

## Acknowledgement
This code refers to the following two projects:

[1] https://github.com/yzyouzhang/AIR-ASVspoof

[2] https://github.com/clovaai/aasist
