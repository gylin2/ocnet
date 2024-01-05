import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from evaluate_tDCF_asvspoof21 import evaluate_tDCF_asvspoof21
from tqdm import tqdm
import eval_metrics as em
import numpy as np
from datawave import ASVspoof2021                    


def test_model(model_path, device):
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

    if args.access_type=='LA':
        path_to_audio = args.path + 'tar_files/ASVspoof2021_LA_eval/flac/'
        path_to_protocol = args.path + 'tar_files/ASVspoof2021_LA_eval/keys/CM/trial_metadata.txt'
        txt_name = 'checkpoint_cm_score_2021.txt'
    elif args.access_type=='DF':
        path_to_audio = args.path + 'tar_files/ASVspoof2021_DF_eval/flac/'
        path_to_protocol = args.path + 'tar_files/ASVspoof2021_DF_eval/keys/CM/trial_metadata.txt'
        txt_name = 'checkpoint_cm_score_2021_DF.txt'
    tag = ["eval", "progress", "hidden_track"]


    test_set = ASVspoof2021(args.access_type, path_to_audio, path_to_protocol, feat_len=args.feat_len, padding="repeat")
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=test_set.collate_fn)                             
    
    model.eval()
    with open(os.path.join(dir_path, txt_name), 'w') as cm_score_file:
        with torch.no_grad():
            for i, (wave, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
                wave = wave.float().to(device)
                tags = tags.to(device)
                labels = labels.to(device)
                score = model(wave).float()

                for j in range(labels.size(0)):
                    cm_score_file.write(
                        '%s %s %s %s\n' % (audio_fn[j], tag[tags[j].data],
                                            "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                            score[j].item()))
    if args.access_type=='DF':
        cm_score_file = os.path.join(dir_path, txt_name)
        cm_data = np.genfromtxt(cm_score_file, dtype=str)
        cm_sources = cm_data[:, 1]
        cm_keys = cm_data[:, 2]
        cm_scores = cm_data[:, 3].astype(np.float)

        cm_keys = cm_keys[cm_sources == args.phase]
        cm_scores = cm_scores[cm_sources == args.phase]

        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_keys == 'spoof']
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
        print("DF EER: {}".format(eer_cm))
    elif args.access_type=='LA':
        asv_score_file = args.path + 'tar_files/ASVspoof2021_LA_eval/keys/'
        evaluate_tDCF_asvspoof21(os.path.join(args.model_dir, txt_name),
                                        asv_score_file, False, args.phase)


def test(model_dir, device):
    model_path = os.path.join(model_dir, "checkpoint/anti-spoofing_model_last.pt")
    # model_path = os.path.join(model_dir, "checkpoint/anti-spoofing_model_best.pt")
    test_model(model_path, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--path", type=str, help="The path of ASVspoof2021",
                        default='/path/ASVspoof2021')
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/OCNet")
    parser.add_argument("-a", "--access_type", type=str, help="LA or DF", default='LA')
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument("--feat_len", type=int, help="features length", default=128000)
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--phase', type=str, default='eval', choices=['progress','eval', 'hidden_track'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test(args.model_dir, args.device)

    # if args.access_type=='LA':
    #     asv_score_file = args.path + 'tar_files/ASVspoof2021_LA_eval/keys/'
    #     evaluate_tDCF_asvspoof21(os.path.join('./models/OCNet', 'checkpoint_cm_score_2021.txt'),
    #                                             asv_score_file, False, args.phase)
    
    # if args.access_type=='DF':
    #     cm_score_file = os.path.join('./models/OCNet', 'checkpoint_cm_score_2021_DF.txt')
    #     cm_data = np.genfromtxt(cm_score_file, dtype=str)
    #     cm_sources = cm_data[:, 1]
    #     cm_keys = cm_data[:, 2]
    #     cm_scores = cm_data[:, 3].astype(np.float)

    #     cm_keys = cm_keys[cm_sources == args.phase]
    #     cm_scores = cm_scores[cm_sources == args.phase]

    #     bona_cm = cm_scores[cm_keys == 'bonafide']
    #     spoof_cm = cm_scores[cm_keys == 'spoof']
    #     eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    #     print("DF EER: {}".format(eer_cm))