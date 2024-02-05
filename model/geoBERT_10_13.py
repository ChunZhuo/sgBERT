import torch
import math
import pytorch_lightning as pl
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce

class local_Module(pl.LightningModule):
    def __init__(self, 
                 MI_num = 4
                   ):
        super().__init__()
        #MI part
        self.MI_num = MI_num
        self.local_Att_mi = local_Att(4,4,4,16)
        self.norm_MI1 = nn.LayerNorm(MI_num)
        self.norm_MI2 = nn.LayerNorm(MI_num)
        self.to_mi_logits = nn.Linear(4,4)

    def forward(self,mi):
        #cross attention
        output_mi = self.local_Att_mi(mi) + mi 
        output_mi = self.norm_MI1(output_mi)
        local_output_mi = self.to_mi_logits(output_mi) + output_mi
        local_output_mi = self.norm_MI2(local_output_mi) 
        return local_output_mi
    
    def embedding(self, mi):
        #cross attention
        output_mi = self.local_Att_mi(mi) + mi 
        output_mi = self.norm_MI1(output_mi)
        local_output_mi = self.to_mi_logits(output_mi) + output_mi
        mi_embedding = self.norm_MI2(local_output_mi) 
        return mi_embedding

class local_Att(pl.LightningModule):
    # this is the attention for mi and sequence 
    def __init__(self, 
                 dim_out,
                 dim_input,
                 dim_head,
                 num_head
                 ):
        super().__init__()
        self.hidden_dim = dim_head*num_head
        self.dim_head = dim_head
        self.dim_k = dim_input
        self.dim_out = dim_out
        self.to_q = nn.Linear(dim_input, self.hidden_dim, bias = False)
        self.to_k = nn.Linear(dim_input, self.hidden_dim , bias = False)
        self.to_v = nn.Linear(dim_input, self.hidden_dim , bias = False)
        self.scale = dim_head ** -0.5
        self.to_out = nn.Linear(self.hidden_dim, self.dim_out)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        
    def forward(self,q_vector):
        d = self.dim_head
        q = self.to_q(q_vector)
        k = self.to_k(q_vector)
        v = self.to_v(q_vector)
        q,k,v = map(lambda t: rearrange(t, 'b l (n d) -> b n l d', d = d), (q,k,v) )
        info = einsum('b n l d, b n L d -> b n l L ', q,k) * self.scale
        info = info.softmax(dim = -1)
        att_out = einsum('b n l L, b n L m -> b n l m', info,v )
        att_out = rearrange(att_out, 'b n l m -> b l (n m)')       
        att_out = self.to_out(att_out) 
        return   att_out

class Whole_Module(pl.LightningModule):
    def __init__(self,
                 num_block = 3,
                 mask_mi_prob = 0.05,
                 MI_num = 4
                 ):
        super().__init__()
        self.MI_num = MI_num
        self.block = nn.ModuleList([local_Module(4) for i in range(num_block)]) 
        self.mi_dense = nn.Linear(self.MI_num,self.MI_num)

        #mask
        self.mask_mi_prob = mask_mi_prob

        #loss
        self.mi_loss = nn.MSELoss()

    def mask(self,mi):
        #seqs,mis are batches of seqs and mi 
        device = mi.device
        batch = mi.shape[0]
        seq_len = mi.shape[1]
        mask_num = math.ceil(0.05 * seq_len)
        mask = torch.zeros(batch,seq_len)
        rand = torch.rand(batch,seq_len)
        _, sampled_indices = rand.topk(mask_num,dim = -1)
        mask = mask.scatter_(-1, sampled_indices, 1).bool().to(device)
        return  mask
    
    def forward(self,mi):
        '''
        mi: moment_invariants
        '''
        mask = self.mask(mi)
        mask_mi = mask.unsqueeze(-1).repeat(1,1,self.MI_num)

        mi_masked = mi.masked_fill(mask_mi, 18)
        

        for block in self.block:
            mi_masked = block(mi_masked)
        mi_output =  self.mi_dense(mi_masked)
        return mi_output

    def embedding(self,mi):
        count = 0 
        for block in self.block:
            if count ==2:
                mi = block.embedding(mi)
            else:
                mi = block(mi)
            count += 1
        return mi
    
    def training_step(self,batch):#,batch_idx):
        mi = batch
        mask_posi = self.mask(mi)
        mi_re = self.forward(mi) 

        loss_train_mi = self.mi_loss(mi_re[mask_posi], mi[mask_posi].to(torch.float16)) 
        train_loss = loss_train_mi 
        self.log('training loss',train_loss)
        return train_loss

    def validation_step(self,batch,batch_idx):
        mi = batch
        mask_posi = self.mask(mi)
        mi_re = self.forward(mi) 
        loss_valid_mi = self.mi_loss(mi_re[mask_posi], mi[mask_posi].to(torch.float16)) 
        valid_loss = loss_valid_mi 
        self.log('valid loss',valid_loss)
        return valid_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 1e-5)
        return optimizer



