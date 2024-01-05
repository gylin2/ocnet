import argparse
import os
import json
import shutil
from utils import setup_seed, L2_regularization
from loss import *
from datawave import ASVspoof2019
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import RawNet
import torch.nn.functional as F

torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True)



def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')                  
    parser.add_argument("-p", "--path", type=str, help="ASVspoof path",
                        default='/path/ASVspoof2019')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/TEMP')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=128000)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=0)

    parser.add_argument('--loss', type=str, default="tocsoftmax",
                        choices=['ce','ocsoftmax', 'tocsoftmax'], help="loss for one-class learning")
    parser.add_argument('--warmup', action='store_true', help="warmup")
    
    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)
    
    # Path for output data
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    else:
        shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
        os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
        file.write("Start recording training loss ...\n")
    with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
        file.write("Start recording validation loss ...\n")

    # assign device
    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args

def adjust_learning_rate(args, optimizer, epoch_num, n_current_steps=0):
    if args.warmup:
        n_warmup_steps=1000
        lr = np.power(64, -0.5) * np.min([
                np.power(n_current_steps, -0.5),
                np.power(n_warmup_steps, -1.5) * n_current_steps])
    else:
        lr = args.lr * (0.5 ** (epoch_num // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    batch_size = args.batch_size
    if args.loss == "ce":
        model = RawNet(args.device, binary_class=True)
    else:
        model = RawNet(args.device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(args.device)
    
    if args.warmup:
        wave_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, 
                                      weight_decay=0, amsgrad=True)
    else:
        wave_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        betas=(args.beta_1, args.beta_2), eps=args.eps, 
                                        weight_decay=0.0005)
    
    path_to_train = args.path + '/'+args.access_type+'/ASVspoof2019_'+args.access_type+'_train/flac/'
    path_to_dev = args.path + '/'+args.access_type+'/ASVspoof2019_'+args.access_type+'_dev/flac/'
    path_to_protocol = args.path + '/'+args.access_type+'/ASVspoof2019_'+args.access_type+'_cm_protocols/'

    training_set = ASVspoof2019(args.access_type, path_to_train, path_to_protocol, 'train', feat_len=args.feat_len, padding=args.padding)
    validation_set = ASVspoof2019(args.access_type, path_to_dev, path_to_protocol, 'dev', feat_len=args.feat_len, padding=args.padding)                        
    trainDataLoader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,
                                 collate_fn=training_set.collate_fn)
    valDataLoader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,
                               collate_fn=validation_set.collate_fn)

    feat, _, _, _ = training_set[29]
    print("Feature shape", feat.shape)
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss().to(args.device)
    if args.loss == "ocsoftmax":
        criterion = OCSoftmax().to(args.device)
    if args.loss == "tocsoftmax":
        criterion = TOCSoftmax().to(args.device)
    
    prev_eer = 1e8
    n_current_steps = 0
    best_epoch = 0
    delta = 1
    for epoch_num in tqdm(range(args.num_epochs)):
        model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)

        if not args.warmup:
            adjust_learning_rate(args, wave_optimizer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        idx_loader, score_loader = [], []
        for i, (wave, _, _, labels) in enumerate(tqdm(trainDataLoader)):
            n_current_steps += delta
            wave = wave.float().to(args.device) 
            labels = labels.to(args.device)
            wave_outputs = model(wave).float()
            if args.loss == "ce":
                wave_loss = criterion(wave_outputs, labels) + L2_regularization(model, 1e-4)
                wave_outputs = F.softmax(wave_outputs)[:,0].float()
            else:
                wave_loss = criterion(wave_outputs, labels) + L2_regularization(model, 1e-4)
            idx_loader.append(labels)
            score_loader.append(wave_outputs.squeeze())

            wave_optimizer.zero_grad()
            trainlossDict[args.loss].append(wave_loss.item())
            wave_loss.backward()
            wave_optimizer.step()
            
            if args.warmup:
                adjust_learning_rate(args, wave_optimizer, epoch_num,n_current_steps)
            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(np.nanmean(trainlossDict[args.loss])) + "\n")

        # Val the model
        model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (wave, _, _, labels) in enumerate(tqdm(valDataLoader)):
                wave = wave.float().to(args.device)
                labels = labels.to(args.device)
                
                score = model(wave).float()

                if args.loss == "ce":
                    wave_loss = criterion(score.float(), labels) + L2_regularization(model, 1e-4)
                    score = F.softmax(score)[:,0].float()
                else:   
                    clone_scores = score.clone()
                    wave_loss = criterion(clone_scores, labels) + L2_regularization(model, 1e-4)

                devlossDict[args.loss].append(wave_loss.item())
                idx_loader.append(labels)
                score_loader.append(score.squeeze())

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()

            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[args.loss])) + "\t" + str(val_eer) +"\n")
            print("Val EER: {}".format(val_eer))
        
        if epoch_num == args.num_epochs-1:
            torch.save(model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_model_last.pt'))

        if val_eer <= prev_eer:
            # Save the model checkpoint
            torch.save(model, os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_model_best.pt'))
            prev_eer = val_eer
        else:
            if epoch_num - best_epoch > 2:
                delta *= 2
                best_epoch = epoch_num + 1
    return model


if __name__ == "__main__":
    args = initParams()
    train(args)