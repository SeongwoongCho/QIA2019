import random
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from SpecAugment import spec_augment_pytorch

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)
    
def preprocess(spectrogram):
    yS = librosa.util.normalize(spectrogram) ## 각 time step마다 max(abs) 로 normalize (즉 무조건 [-1,1] 범위에 온다.)
    yS = (1+yS)/2
    if yS.shape[1]<224:
        total = 224-yS.shape[1]
        left = total//2
        right = total - left
        yS=np.pad(yS,((0,0),(left,right)),'constant')
    return yS

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def to_onehot(label,num_classes=7):
    return np.eye(num_classes)[label]

def cross_entropy():
    def _cross_entropy(input, target, size_average=True):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean

        Examples::

            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)

            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        logsoftmax = nn.LogSoftmax()
        if size_average:
            return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
        else:
            return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
    return _cross_entropy

def add_noise(spec,power = 0.1):
    return spec + power*np.random.normal(spec)

def augment(yS):
    ## add noise / specaug
    yS = add_noise(yS,power = np.random.uniform(0,0.15))
    try:
        yS = spec_augment_pytorch.spec_augment(torch.Tensor([yS]),time_warping_para=120, frequency_masking_para=27,
                                                 time_masking_para=30, frequency_mask_num=3, time_mask_num=4)
        yS = yS.cpu().detach().numpy()[0]
    except:
        pass
    return yS

### trash can ### 
def mix_db(x,noise,db):
    E_x=np.mean(x**2)
    E_y=np.mean(noise**2)
    a = E_x/(E_y)*(10**(db/10))
    return (x+a*noise)/(a+1)