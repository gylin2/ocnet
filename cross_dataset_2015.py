import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import eval_metrics as em
import numpy as np
from datawave import ASVspoof2015                    

def test_model(model_path, part, device):
    dirname = os.path.dirname
    if "checkpoint" in dirname(model_path):
        dir_path = dirname(dirname(model_path))
    else:
        dir_path = dirname(model_path)

    model = torch.load(model_path, map_location="cuda")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        try:
            # Train on multiple GPUs, test on multiple GPUs
            model = nn.DataParallel(model.module).to(device)
        except:
            # Train on a single GPU, test on multiple GPUs
            model = nn.DataParallel(model).to(device)
    else:
        try:
            # Train on multiple GPUs, test on a single GPU
            model = nn.DataParallel(model.module).to(device)
        except:
            # Train on a single GPU, test on a single GPU
            model = model.to(device)


    path_to_protocol = args.path+'/CM_protocol/cm_evaluation.ndx'
    path_to_audio = args.path+'/wav/'
    test_set = ASVspoof2015(path_to_audio, path_to_protocol, part, feat_len=args.feat_len, padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=test_set.collate_fn)                             
    
    model.eval()
    score_loader, idx_loader = [], []
    with open(os.path.join(dir_path, 'checkpoint_cm_score_2015.txt'), 'w') as cm_score_file:
        with torch.no_grad():
            for i, (wave, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
                wave = wave.float().to(device)
                tags = tags.to(device)
                labels = labels.to(device)
                score = model(wave).float()
            
                for j in range(labels.size(0)):
                    cm_score_file.write(
                        '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                            "spoof" if labels[j].data.cpu().numpy() else "human",
                                            score[j].item()))
                for j in range(labels.size(0)):
                    score_loader.append(score[j].item())
                    idx_loader.append(labels[j].data.cpu().numpy())

    # Load CM scores
    cm_data = np.genfromtxt(os.path.join(dir_path, 'checkpoint_cm_score_2015.txt'), dtype=str)
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)
    other_cm_scores = -cm_scores
    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'human']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'human'], other_cm_scores[cm_keys == 'spoof'])[0]
    eer = min(eer_cm, other_eer_cm)
    print('EER = {:8.5f} % (Equal error rate for countermeasure)'.format(eer * 100))
    
    return eer

def test(model_dir, device):
    model_path = os.path.join(model_dir, "checkpoint/anti-spoofing_model_last.pt")
    # model_path = os.path.join(model_dir, "checkpoint/anti-spoofing_model_best.pt.pt")
    test_model(model_path, args.part, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--path", type=str, help="The path of ASVspoof2019",
                        default='/path/ASVspoof2015')
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/OCNet")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument("--feat_len", type=int, help="features length", default=128000)
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--part', type=str, default='eval', choices=['train','dev', 'eval'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test(args.model_dir, args.device)


    # cm_data = np.genfromtxt(os.path.join(args.model_dir, 'checkpoint_cm_score_2015.txt'), dtype=str)
    # cm_keys = cm_data[:, 2]
    # cm_scores = cm_data[:, 3].astype(np.float)
    # other_cm_scores = -cm_scores
    # # Extract bona fide (real human) and spoof scores from the CM scores
    # bona_cm = cm_scores[cm_keys == 'human']
    # spoof_cm = cm_scores[cm_keys == 'spoof']
    # eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    # other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'human'], other_cm_scores[cm_keys == 'spoof'])[0]
    # eer = min(eer_cm, other_eer_cm)
    # print('EER = {:8.5f} % (Equal error rate for countermeasure)'.format(eer * 100))
