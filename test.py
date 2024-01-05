import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np
from datawave import ASVspoof2019
import torch.nn.functional as F
from utils import setup_seed

                      
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True)


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


    path_to_audio = os.path.join(args.path, args.access_type, 'ASVspoof2019_'+args.access_type +'_'+ part +'/flac/')
    path_to_protocol = args.path+"/"+args.access_type+'/ASVspoof2019_'+args.access_type+"_cm_protocols/"
    test_set = ASVspoof2019(args.access_type, path_to_audio, path_to_protocol, part, feat_len=args.feat_len, padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=test_set.collate_fn)                             

    model.eval()
    if part == 'dev':
        checkpoint_file = 'checkpoint_cm_score_dev.txt'
    else:
        checkpoint_file = 'checkpoint_cm_score.txt'
    with open(os.path.join(dir_path, checkpoint_file), 'w') as cm_score_file:
        with torch.no_grad():
            for i, (wave, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
                wave = wave.float().to(device)
                labels = labels.to(device)

                tags = tags.to(device)
                score = model(wave).float()
                if score.size(-1)>1:
                    score = F.softmax(score)[:,0].float()
                for j in range(labels.size(0)):
                    cm_score_file.write(
                        '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                            "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                            score[j].item()))
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, checkpoint_file), args.path)
    return eer_cm, min_tDCF

def test(model_dir, device):
    model_path = os.path.join(model_dir, "checkpoint/anti-spoofing_model_last.pt")
    # model_path = os.path.join(model_dir, "checkpoint/anti-spoofing_model_best.pt")
    test_model(model_path, args.part, device)


def test_individual_attacks(cm_score_file):
    asv_score_file = os.path.join(args.path,
                                  'LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(7,20):
        # Extract target, nontarget, and spoof scores from the ASV scores
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_sources == 'A%02d' % attack_idx]

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

        if eer_cm < other_eer_cm:
            # Compute t-DCF
            tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,
                                                        True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]

        else:
            tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == 'bonafide'],
                                                        other_cm_scores[cm_sources == 'A%02d' % attack_idx],
                                                        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
        eer_cm_lst.append(min(eer_cm, other_eer_cm))
        min_tDCF_lst.append(min_tDCF)

    return eer_cm_lst, min_tDCF_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-p", "--path", type=str, help="The path of ASVspoof2019",
                        default='/path/ASVspoof2019')
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/Net")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument("--feat_len", type=int, help="features length", default=128000)
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--part', type=str, default='eval', choices=['train','dev', 'eval'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(0)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test(args.model_dir, args.device)
    
    
    
    # eer_cm_lst, min_tDCF_lst = test_individual_attacks(os.path.join(args.model_dir, 'checkpoint_cm_score.txt'))
    # eer_cm_lst = [x*100 for x in eer_cm_lst]
    # print(eer_cm_lst)
    # print(min_tDCF_lst)

    # if args.part == 'dev':
    #     checkpoint_file = 'checkpoint_cm_score_dev.txt'
    # else:
    #     checkpoint_file = 'checkpoint_cm_score.txt'
    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(args.model_dir, checkpoint_file), args.path)

    