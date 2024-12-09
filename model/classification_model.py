import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.downsample = nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=2, stride=2, bias=False) if downsample else None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        identity = self.shortcut(x)

        out = out+identity
        out = self.relu(out)

        if self.downsample:
            out = self.downsample(out)
        
        return out


class TumorClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block(1, 64, 3, downsample=None)
        self.block2 = Block(64, 128, 5, downsample=2)
        self.block3 = Block(128, 128, 7, downsample=2)
        self.block4 = Block(128, 128, 5, downsample=2)
        self.block5 = Block(128, 128, 3, downsample=2)
        self.block6 = Block(128, 128, 3, downsample=2)
        
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def extract_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        feats = self.block6(x)
        
        x = torch.flatten(feats, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x, feats
