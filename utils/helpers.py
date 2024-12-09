import glob
import torch
import os
import os.path as osp
import torch.nn as nn

def load_weights(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def save_weights(model, name, work_dir, rem_prev=None):
    if rem_prev is not None:
        pre_results = glob.glob(osp.join(work_dir, f"{rem_prev}*.pth"))
        assert len(pre_results)<=1
        if len(pre_results)==1:
            os.remove(pre_results[0])
    torch.save(model.state_dict(), osp.join(work_dir, name))

def orthogonal_init(layer):
    for name, param in layer.named_parameters():
        if 'weight_hh' in name:
            nn.init.orthogonal_(param)

def initialize_forget_gate(layer):
    for name, param in layer.named_parameters():
        if 'bias_hh' in name:
            n = param.size(0)
            param.data[n//4:n//2].fill_(1.0)

def get_current_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]
