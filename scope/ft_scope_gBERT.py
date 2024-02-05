import torch 
from torch import nn 
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
sys.path.append('/lustre/BIF/nobackup/zhang408/Lightning/Transformer/geoBERT/')


class scope(pl.LightningModule):
    #model: model name, string 
    #freeze: boolean value 
    def __init__(self, num_feature, num_class,freeze):
        super().__init__()
        self.pretrain = torch.load(f"/lustre/BIF/nobackup/zhang408/Lightning/Transformer/geoBERT/gBERT.pt")
        if freeze:
            self.pretrain.eval()
        self.Linear = nn.Linear(num_feature, num_class)
        self.softmax = nn.Softmax(dim =1)

    def forward(self, mi):
        mi_embedding = self.pretrain.embedding(mi)
        mi_embedding = mi_embedding.permute(0,2,1).mean(2)
        logit = self.Linear(mi_embedding)
        return logit
     
    def training_step(self,batch,batch_idx):
        mi,seq, y = batch
        logit = self.forward(mi)
        train_loss = F.cross_entropy(logit, y)
        self.log('training loss',train_loss)
        return train_loss
    
    def validation_step(self,batch,batch_idx):
        mi,seq,y = batch
        logit = self.forward(mi)
        valid_loss = F.cross_entropy(logit, y)
        self.log('valid loss',valid_loss)
        return valid_loss
    
    def test_step(self,batch):
        mi,seq,y = batch
        logit = self.forward(mi)
        output = self.softmax(logit)
        predict = torch.argmax(output,dim =1)
        return predict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 1e-3)
        return optimizer


    
    