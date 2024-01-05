import numpy as np
import os
import soundfile as sf
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
torch.set_default_tensor_type(torch.FloatTensor)


class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_audio, path_to_protocol, part='train', feat_len=128000, padding='repeat'):
        self.access_type = access_type
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.part = part
        self.feat_len = feat_len
        self.padding = padding

        if self.part == 'train':
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trn.txt')
        else:
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')

        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

            print(len(self.all_info))
    
    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        _, filename, _, tag, label = self.all_info[idx]
        wav, sr = sf.read(self.path_to_audio+filename+'.flac')

        wav = torch.from_numpy(wav).unsqueeze(0)
        
        this_feat_len = wav.shape[1]
        if this_feat_len > self.feat_len:
            if self.part in ["eval", "dev"]:
                wav = wav[:, :self.feat_len]
            else:
                startp = np.random.randint(this_feat_len-self.feat_len)
                wav = wav[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                wav = padding(wav, self.feat_len)
            elif self.padding == 'repeat':
                wav = repeat_padding(wav, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return wav.squeeze(0), filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2015(Dataset):
    def __init__(self, path_to_audio, path_to_protocol, part='eval', feat_len=128000, padding='repeat'):
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.part = part
        self.feat_len = feat_len
        self.padding = padding

        self.tag = {"human": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
                    "S6": 6, "S7": 7, "S8": 8, "S9": 9, "S10": 10}
        self.label = {"spoof": 1, "human": 0}

        with open(self.path_to_protocol, 'r') as f:
            self.audio_info = [info.strip().split() for info in f.readlines()]

    def __len__(self):
        return len(self.audio_info)

    def __getitem__(self, idx):
        files, filename, tag, label = self.audio_info[idx]
        wav, sr = sf.read(self.path_to_audio+files+'/'+filename+'.wav')
        wav = torch.from_numpy(wav).unsqueeze(0)
        
        this_feat_len = wav.shape[1]
        if this_feat_len > self.feat_len:
            if self.part in ["eval", "dev"]:
                wav = wav[:, :self.feat_len]
            else:
                startp = np.random.randint(this_feat_len-self.feat_len)
                wav = wav[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                wav = padding(wav, self.feat_len)
            elif self.padding == 'repeat':
                wav = repeat_padding(wav, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')
            
        return wav.squeeze(0), filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)

class ASVspoof2021(Dataset):
    def __init__(self, access_type, path_to_audio, path_to_protocol, feat_len=128000, padding='repeat'):
        self.access_type = access_type
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.feat_len = feat_len
        self.padding = padding
        
        self.tag = {"eval": 0, "progress": 1, "hidden_track": 2}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info
            print(len(self.all_info))

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        _, filename, _, _,_, label,_,tag = self.all_info[idx]
        wav, sr = sf.read(self.path_to_audio+filename+'.flac')
        wav = torch.from_numpy(wav).unsqueeze(0)
        
        this_feat_len = wav.shape[1]
        if this_feat_len > self.feat_len:
            wav = wav[:, :self.feat_len]

        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                wav = padding(wav, self.feat_len)
            elif self.padding == 'repeat':
                wav = repeat_padding(wav, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')
            
        return wav.squeeze(0), filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)           

def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


if __name__ == "__main__":
    path_to_audio = '/path/ASVspoof2019/LA/'


