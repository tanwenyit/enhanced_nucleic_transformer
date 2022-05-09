import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from Functions import *
import matplotlib.pyplot as plt

tokens='ACGU().BEHIMSX'
#eterna,'nupack','rnastructure','vienna_2','contrafold_2',
class RNADataset(Dataset):
    def __init__(self,seqs,labels,ids,ew,bpp_path,transform=None,training=True):
        self.transform=transform
        self.seqs=seqs#.transpose(1,0,2,3)
        #print(self.data.shape)
        self.data=[]
        self.labels=labels.astype('float32')
        self.bpp_path=bpp_path
        self.ids=ids
        self.ew=ew
        self.training=training
        self.bpps=[]
        # dm is distance matrix
        dm=get_distance_mask(len(seqs[0]))#.reshape(1,bpps.shape[-1],bpps.shape[-1])
        # print(dm.shape)
        # exit()
        for i,id in tqdm(enumerate(self.ids)):
            bpps=np.load(os.path.join(self.bpp_path,'train_test_bpps',id+'_bpp.npy'))
            dms=np.asarray([dm for i in range(bpps.shape[0])])
            
            # concatenate base pairing matrix (bpp) with distance matrix (dm)
            bpps=np.concatenate([bpps.reshape(bpps.shape[0],1,bpps.shape[1],bpps.shape[2]),dms],1) # (12,4,130,130)

            with open(os.path.join(self.bpp_path,'train_test_bpps',id+'_struc.p'),'rb') as f:
                structures=pickle.load(f)
            with open(os.path.join(self.bpp_path,'train_test_bpps',id+'_loop.p'),'rb') as f:
                loops=pickle.load(f)
            seq=self.seqs[i]
            # print(seq)
            # exit()
            input=[]
            # bpps.shape[0] = 12
            for j in range(bpps.shape[0]):
                input_seq=np.asarray([tokens.index(s) for s in seq])
                input_structure=np.asarray([tokens.index(s) for s in structures[j]])
                input_loop=np.asarray([tokens.index(s) for s in loops[j]])
                input.append(np.stack([input_seq,input_structure,input_loop],-1)) # np.stack(..., axis=-1) is stack then transpose, so ->(12, len(seq), 3)
            input=np.asarray(input).astype('int')
            #print(input.shape)
            self.data.append(input) # (sample_no, 12, len(seq), 3)
            #exit()
                #print(np.stack([input_seq,input_structure,input_loop],-1).shape)
                #exit()
                # plt.subplot(1,4,1)
                # for _ in range(4):
                #     plt.subplot(1,4,_+1)
                #     plt.imshow(bpps[0,_])
                # plt.show()
                # exit()
            self.bpps.append(np.clip(bpps,0,1).astype('float32')) # np.clip() is used to limit the value in between self-determined min and max
        self.data=np.asarray(self.data)

    # Use for DataLoader
    def __len__(self):
        return len(self.data)
    
    # Use for DataLoader
    def __getitem__(self, idx):
        #sample = {'data': self.data[idx], 'labels': self.labels[idx]}
        if self.training:
            bpp_selection=np.random.randint(self.bpps[idx].shape[0])
            #print(self.bpps[idx].shape[0])
            sample = {'data': self.data[idx][bpp_selection], 'labels': self.labels[idx], 'bpp': self.bpps[idx][bpp_selection],
            'ew': self.ew[idx],'id':self.ids[idx]}
        else:
            sample = {'data': self.data[idx], 'labels': self.labels[idx], 'bpp': self.bpps[idx],
            'ew': self.ew[idx],'id':self.ids[idx]}
        if self.transform:
            sample=self.transform(sample)
        return sample
