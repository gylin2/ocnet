import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=7, padding=0, dilation=1, bias=False, groups=1,freq_scale='Mel'):
        super(SincConv,self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)


        if freq_scale == 'Mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.freq=filbandwidthsf[:self.out_channels]

        elif freq_scale == 'Inverse-mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.mel=filbandwidthsf[:self.out_channels]
            self.freq=np.abs(np.flip(self.mel)-1) ## invert mel scale
        else:
            fmelmax=np.max(f)
            fmelmin=np.min(f)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
            self.freq=filbandwidthsmel
        
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels,self.kernel_size)
    
        
    def forward(self,x):
        for i in range(len(self.freq)-1):
            fmin=self.freq[i]
            fmax=self.freq[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp.cpu()/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp.cpu()/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Residual_block(nn.Module):
    def __init__(self, nb_filts):
        super(Residual_block, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(num_features = nb_filts[1])
        
        self.selu = nn.SELU(inplace=True)

        self.se = SELayer(nb_filts[1], reduction=4)
        self.conv1 = nn.Conv2d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = (2,3),
			padding = (1,1),
			stride = 1,
            bias=False)
        
        self.bn2 = nn.BatchNorm2d(num_features = nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			kernel_size = (2,3),
            padding = (0,1),
			stride = 1,
            bias=False)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = (0,1),
				kernel_size = (1,3),
				stride = 1,
                bias=False)  
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1,3)) 
        
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.selu(out)
        out = self.mp(out)
        return out

class RawNet(nn.Module):
    def __init__(self, device, binary_class=False):
        super(RawNet, self).__init__()
        self.device=device

        self.Sinc_conv=SincConv(device=self.device,
			out_channels = 70,
			kernel_size = 128,
            in_channels = 1,
            freq_scale='Linear'
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.block0 = nn.Sequential(Residual_block(nb_filts = [1, 32]))
        self.block1 = nn.Sequential(Residual_block(nb_filts = [32, 32]))
        self.block2 = nn.Sequential(Residual_block(nb_filts = [32, 24]))
        self.block3 = nn.Sequential(Residual_block(nb_filts = [24, 24]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = [24, 24]))

        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.bn_before_gru = nn.BatchNorm2d(num_features = 24)

        # OC-Softmax and TOC-Softmax
        if binary_class:
            self.fc = nn.Linear(24, 2)
        else:
            self.fc = NormedLinear(24, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.Sinc_conv(x)    # Fixed sinc filters convolution
        x = x.unsqueeze(dim=1)

        # x = torchaudio.transforms.AmplitudeToDB(stype='amplitude')(x).float() # Added for PA attacks

        x = F.max_pool2d(torch.abs(x), (1, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.bn_before_gru(x)

        # low-fre
        x = x[:,:,:int(x.shape[2]/2),:].contiguous()


        # directed statistics pooling
        mu = torch.mean(x, dim=2)
        feat = torch.std(mu, dim=-1)

        # x = self.avgpool(x).view(x.size()[0], -1)

        x = self.fc(feat)
        return x