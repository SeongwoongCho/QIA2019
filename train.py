#data Augmentation
import numpy as np
import os
import multiprocessing as mp

from utils import *
from dataloader import *

import time
import csv
import warnings

##training
import torch
import torch.nn as nn
from torch import optim
from radam import RAdam,Lookahead
from apex import amp

##model
from efficientnet_pytorch import EfficientNet

##UI
from tqdm import tqdm

warnings.filterwarnings('ignore') ## ignore torch warning
seed_everything(42)

model_name = 'b0'
n_classes = 7
learning_rate=6e-3
alpha = 0.2
smooth_weight = 0
n_epoch = 100
batch_size = 256

num_workers= mp.cpu_count()
save_path="./model/{}_smooth_{}_mix_{}_RANGER_COSINEANNEALING/".format(model_name,smooth_weight,alpha)
os.makedirs(save_path,exist_ok=True)

train_dir='../audioData/train_process_mel/'
valid_dir='../audioData/val_process_mel/'
train_list=os.listdir(train_dir)
valid_list=os.listdir(valid_dir)
train_list = [item for item in train_list if '.npy' in item]
valid_list = [item for item in valid_list if '.npy' in item]

train_dataset = Dataset(file_list=train_list,root_dir=train_dir,label_smooth_weight=smooth_weight,is_train=True)
valid_dataset = Dataset(file_list=valid_list,root_dir=valid_dir,label_smooth_weight=0,is_train=False)
train_loader=data.DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,pin_memory=True)
valid_loader=data.DataLoader(dataset=valid_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,pin_memory=True)
                
model = EfficientNet.from_pretrained(model_name='efficientnet-'+model_name)
model._fc = nn.Linear(model._fc.in_features,n_classes)
criterion = cross_entropy()
model.cuda()

print("train_loader length :",len(train_loader))

optimizer = RAdam(model.parameters(),lr=learning_rate,weight_decay = 5e-5)
optimizer = Lookahead(optimizer, alpha=0.5, k=6)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = nn.DataParallel(model)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = n_epoch*len(train_loader), eta_min=learning_rate/20)

print("training....")
log=open(os.path.join(save_path,'log.csv'),'a+', encoding='utf-8',newline='')
log_writer=csv.writer(log)
log_writer.writerow(["epoch","train_loss","train_acc","valid_loss","valid_acc"])
log.flush()

best_val=np.inf
for epoch in range(n_epoch):
    train_loss=0
    train_acc=0
    
    optimizer.zero_grad()
    model.train()
    for idx,(x_train,y_train) in enumerate(tqdm(train_loader)):
        x_train,y_train=x_train.cuda(),y_train.cuda()
        x_train,y_train_a,y_train_b,lam = mixup_data(x_train,y_train,alpha)
        pred = model(x_train)
        loss = criterion(pred,y_train_a*lam + y_train_b*(1-lam))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss+=loss.item()/len(train_loader)
        pred=pred.detach().max(1)[1]
        y_train_a = y_train_a.detach().max(1)[1]
        y_train_b = y_train_b.detach().max(1)[1]
        train_acc += (lam * pred.eq(y_train_a.view_as(pred)).sum().item() + (1-lam)* pred.eq(y_train_b.view_as(pred)).sum().item())
        scheduler.step()
    train_acc/=len(train_list)
    
    valid_loss=0
    valid_acc=0
    model.eval()
    for idx,(x_val,y_val) in enumerate(tqdm(valid_loader)):
        x_val,y_val=x_val.cuda(),y_val.cuda()
        with torch.no_grad():
            pred = model(x_val)
            loss = criterion(pred,y_val)
        valid_loss+=loss.item()/len(valid_loader)
            
        pred=pred.detach().max(1)[1]
        y_val = y_val.detach().max(1)[1]
        valid_acc+=pred.eq(y_val.view_as(pred)).sum().item()
    valid_acc/=len(valid_list)

    torch.save(model.state_dict(),os.path.join(save_path,'%d_best_%.4f.pth'%(epoch,valid_loss)))
    log_writer.writerow([epoch,train_loss,train_acc,valid_loss,valid_acc])
    log.flush()
    print("Epoch [%d]/[%d] train_loss: %.6f train_Acc: %.6f valid_loss: %.6f valid_acc:%.6f"%(
    epoch,n_epoch,train_loss,train_acc,valid_loss,valid_acc))
    
log.close()