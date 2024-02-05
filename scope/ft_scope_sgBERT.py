import torch 
from torch import nn 
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
sys.path.append('/lustre/BIF/nobackup/zhang408/Lightning/Transformer/geoBERT/')
#from geoBERT_10_5 import Whole_Module

class scope(pl.LightningModule):
    #model: model name, string 
    #freeze: boolean value 
    def __init__(self, num_feature, num_class,freeze,add_seq):
        super().__init__()
        self.add_seq = add_seq
        self.pretrain = torch.load(f"/lustre/BIF/nobackup/zhang408/Lightning/Transformer/geoBERT/sgBERT.pt")
        if freeze:
            self.pretrain.eval()
        if add_seq == False:
            self.Linear = nn.Linear(num_feature, num_class)
        else:
            self.Linear = nn.Linear(num_feature+16,num_class)
        self.softmax = nn.Softmax(dim =1)

    def forward(self, mi,seq):
        mi_embedding,seq_embedding = self.pretrain.embedding(mi,seq)

        if self.add_seq == True:
            mi_embedding = mi_embedding.permute(0,2,1).mean(2)
            seq_embedding = seq_embedding.permute(0,2,1).mean(2)
            embedding = torch.cat((mi_embedding, seq_embedding),1)
        else:    
            embedding = mi_embedding.permute(0,2,1).mean(2)
        #print(embedding.shape)
        logit = self.Linear(embedding)
        return logit
     
    def training_step(self,batch,batch_idx):
        mi,seq, y = batch
        logit = self.forward(mi,seq)
        train_loss = F.cross_entropy(logit, y)
        self.log('training loss',train_loss)
        return train_loss
    
    def validation_step(self,batch,batch_idx):
        mi,seq,y = batch
        logit = self.forward(mi,seq)
        valid_loss = F.cross_entropy(logit, y)
        self.log('valid loss',valid_loss)
        return valid_loss
    
    def test_step(self,batch):
        mi,seq,y = batch
        logit = self.forward(mi,seq)
        output = self.softmax(logit)
        predict = torch.argmax(output,dim =1)
        return predict
    
    def MHA_invest(self,batch):
        mi,seq,y = batch
        logit = self.forward(mi,seq)
        output = self.softmax(logit)
        predict = torch.argmax(output,dim =1)
        if predict == y:
            print("prediction success!")
        Att = self.pretrain.MHA_invest(mi,seq)
        return Att 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 1e-3)
        return optimizer


    
    