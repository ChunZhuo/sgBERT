import torch
import pytorch_lightning as pl
import numpy as np
from typing import Optional
import  torch.utils.data as data 
from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os.path

class proteinDM(pl.LightningDataModule):
    def __init__(self, data_dir,  
            batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def subdata(self,fl_mi, fl_label, fl_seq):
        '''
        get moment invariants from file
        '''
        labels = {}
        with open(self.data_dir + fl_label, "r") as fl:
            lines_l = fl.readlines()
        for line in lines_l:
            id = line.split(" ")[0]
            label = line.split(" ")[1][-2]
            labels[id] =  label
        Seqs = {}
        with open(self.data_dir + fl_seq, "r") as fs:
            lines_s = fs.readlines()
        for line in lines_s:
                if line.startswith(">"):
                    id = line.split(" ")[0][1:5]
                    Seqs[id] = ""
                else:
                    Seqs[id] = Seqs[id]+ line.rstrip("\n") 
                     
        AA = {'A':1,'R':2,'N':3,'D':4,'C':5,
              'Q':6,'E':7,'G':8,'H':9,
              'I':10,'L':11,'K':12,'M':13,
              'F':14,'P':15,'S':16,'T':17, 'W':18,
              'Y':19,'V':20, 'U':21, 'O': 22, 'X':23,
              'B':24,'Z':25, 'J':26}
        
        seq_tokens = {}
        for id in Seqs:
            seq_tokens[id] = []
            for aa in Seqs[id]:
                seq_tokens[id].append(AA[aa])

        MI = {}
        with open(self.data_dir + fl_mi, "r") as fp:
            lines_p = fp.readlines()  
        for line in lines_p:
            if line.startswith(">"):
                ID = line.rstrip('\n')[1:]
                MI[ID] = []
            else:
                MI[ID].append([float(i) for i in line.rstrip().split(',')])

        data_ready = {}

        for id in MI:
            Len = len(seq_tokens[id])
            if len(MI[id][0:Len]) == len(seq_tokens[id]):
                data_ready[id] = (MI[id][0:Len], seq_tokens[id], labels[id]) 
            
        return data_ready

    def setup(self, stage:Optional[str] = None):
        torch.manual_seed(0)
        if not os.path.exists('input_list_10000.pth'):
            data_ready = self.subdata("sample_MI_10000","pdb_classes_10000","samples_10000.fasta")
            IDs = list(data_ready.keys())
            
            self.dataset = []
            for id in IDs:
                protein = torch.tensor(data_ready[id][0])
                seq = torch.tensor(data_ready[id][1])
                label = torch.tensor(int(data_ready[id][2]))
                self.dataset.append((protein,seq,torch.tensor(label)))
            torch.save(self.dataset ,"input_list_10000.pth")
        else:
            self.dataset = torch.load('input_list_10000.pth')
        train_size = int(len(self.dataset)*0.8)
        valid_size = len(self.dataset) - train_size  
        self.train,self.valid = data.random_split(self.dataset, [train_size, valid_size])  

    def padding(self,batch):
        # this is the maximum padding
        
        mi = pad_sequence([item[0] for item in batch])
        seq = pad_sequence([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        mi = mi.permute(1,0,2)
        seq = seq.permute(1,0)
        return mi,seq,labels
    
    def train_dataloader(self):
        return data.DataLoader(
                self.dataset, batch_size = self.batch_size, shuffle = False,num_workers = 4,collate_fn = self.padding)
    
    def val_dataloader(self):
        return data.DataLoader(
                self.valid, batch_size= self.batch_size, shuffle = False,num_workers = 4,collate_fn = self.padding)

    def test_dataloader(self):
        return data.DataLoader(
                self.test, batch_size= self.batch_size, shuffle = False,num_workers = 4,collate_fn = self.padding)
        