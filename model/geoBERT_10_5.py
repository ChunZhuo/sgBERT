import math
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, einsum
import einops
from einops import rearrange, reduce

class block_Module(pl.LightningModule):
    def __init__(self, 
                 MI_num = 4,
                 seq_embedding_size = 16,
                 dim_head = 16,
                 num_head = 4
                   ):
        super().__init__()

        #MI part
        self.MI_num = MI_num

        self.local_Att_mi = local_Att(MI_num,seq_embedding_size,MI_num,dim_head,num_head)
        self.local_Att_seq = local_Att(seq_embedding_size,seq_embedding_size,seq_embedding_size,dim_head, num_head)
        self.norm_seq1 = nn.LayerNorm(seq_embedding_size)
        self.norm_seq2 = nn.LayerNorm(seq_embedding_size)
        self.norm_MI1 = nn.LayerNorm(MI_num)
        self.norm_MI2 = nn.LayerNorm(MI_num)
        self.mi_linear = nn.Linear(4,4)
        self.seq_linear = nn.Linear(16,16)

    def forward(self,mi,seq):
        #cross attention
        output_mi = self.local_Att_mi(seq, mi) + mi 
        output_mi = self.norm_MI1(output_mi)
        local_output_mi = self.mi_linear(output_mi) + output_mi
        local_output_mi = self.norm_MI2(local_output_mi)

        #seq  self-attention
        seq = self.local_Att_seq(seq,seq) + seq
        seq = self.norm_seq1(seq)
        seq_output  = self.seq_linear(seq) + seq
        seq_output = self.norm_seq2(seq_output)
        return local_output_mi,seq_output
    
    def embedding(self, mi, seq):
        #cross attention
        output_mi = self.local_Att_mi(seq, mi) + mi 
        output_mi = self.norm_MI1(output_mi)
        local_output_mi = self.mi_linear(output_mi) + output_mi
        mi_embedding = self.norm_MI2(local_output_mi)

        #seq  self-attention
        seq = self.local_Att_seq(seq,seq) + seq
        seq = self.norm_seq1(seq)
        seq_output  = self.seq_linear(seq) + seq
        seq_embedding = self.norm_seq2(seq_output)
        return mi_embedding,seq_embedding
    
    def MHA_invest(self, mi, seq):
        info = self.local_Att_mi.MHA_invest(seq,mi)
        return info


class local_Att(pl.LightningModule):
    # this is the attention for mi and sequence 
    def __init__(self, 
                 dim_out,
                 dim_q,
                 dim_k,
                 dim_head,
                 num_head
                 ):
        super().__init__()
        self.hidden_dim = dim_head*num_head
        self.dim_head = dim_head
        self.dim_k = dim_k
        self.dim_out = dim_out
        self.to_q = nn.Linear(dim_q, self.hidden_dim, bias = False)
        self.to_k = nn.Linear(dim_k, self.hidden_dim, bias = False)
        self.to_v = nn.Linear(dim_k, self.hidden_dim, bias = False)
        self.scale = math.sqrt(dim_head)
        self.to_out = nn.Linear(self.hidden_dim, self.dim_out)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)

    def forward(self,q_vector,k_vector):
        d = self.dim_head
        q = self.to_q(q_vector)
        k = self.to_k(k_vector)
        v = self.to_v(k_vector)
        q,k,v = map(lambda t: rearrange(t, 'b l (n d) -> b n l d', d = d), (q,k,v) )
        info = einsum('b n l d, b n L d -> b n l L ', q,k) * self.scale
        info = info.softmax(dim = -1)
        att_out = einsum('b n l L, b n L m -> b n l m', info,v )
        att_out = rearrange(att_out, 'b n l m -> b l (n m)')       
        att_out = self.to_out(att_out) 
        return  att_out
    
    def MHA_invest(self, q_vector, k_vector):
        d = self.dim_head
        q = self.to_q(q_vector)
        k = self.to_k(k_vector)
        v = self.to_v(k_vector)
        q,k,v = map(lambda t: rearrange(t, 'b l (n d) -> b n l d', d = d), (q,k,v) )
        info = einsum('b n l d, b n L d -> b n l L ', q,k) * self.scale
        info = info.softmax(dim = -1)
        return info
    
class Whole_Module(pl.LightningModule):
    def __init__(self,
                 num_block = 3,
                 mask_mi_prob = 0.05,
                 mask_seq_prob = 0.05,
                 seq_embedding_size = 16,
                 narrow_kernel = 8,
                 MI_num = 4,
                 
                 ):
        super().__init__()
        self.MI_num = MI_num
        self.block = nn.ModuleList([block_Module(MI_num, seq_embedding_size) for i in range(num_block)]) 
        #mask
        self.mask_mi_prob = mask_mi_prob
        self.mask_seq_prob = mask_seq_prob

        self.seq_embedding_size = seq_embedding_size
        self.seq_embedding = nn.Embedding(26, seq_embedding_size, padding_idx=0)

        self.s_narrow_conv = nn.Sequential(
            nn.Conv1d(seq_embedding_size, seq_embedding_size, narrow_kernel, padding = 3),#(narrow_kernel // 2)),
            nn.GELU()
        ) 
        self.mi_dense = nn.Linear(self.MI_num,MI_num)
        self.seq_dense = nn.Linear(self.seq_embedding_size,26)
        #loss
        self.seq_loss = nn.CrossEntropyLoss() 
        self.mi_loss = nn.MSELoss()
    
    def mask(self,seqs):
        #seqs,mis are batches of seqs and mi 
        device = seqs.device
        batch = seqs.shape[0]
        seq_len = seqs.shape[1]
        mask_num = math.ceil(0.05 * seq_len)
        mask = torch.zeros(batch,seq_len)
        rand = torch.rand(batch,seq_len)
        _, sampled_indices = rand.topk(mask_num,dim = -1)
        mask = mask.scatter_(-1, sampled_indices, 1).bool().to(device)
        return  mask

    def forward(self,mi,seq):
        '''
        mi: moment_invariants
        seq: sequence 
        '''
        
        mask = self.mask(seq)
        #mask_seq = self.mask(seq)

        #mask for mi
        mask_mi = mask.unsqueeze(-1).repeat(1,1,self.MI_num)
        
        #mask for seq
        mask_seq = mask.unsqueeze(-1).repeat(1,1,self.seq_embedding_size)
        
        ## MI part
        ## mask token for MI : 18
        mi_masked = mi.masked_fill(mask_mi, 18)

        # this zero_padding is to make sure after convlution, seq length remains the same
        device = seq.device
        batch_size = seq.shape[0]
        zero_padding = torch.zeros((batch_size,1),dtype = seq.dtype).to(device)
        seq = torch.cat((seq, zero_padding),1)
        mask_seq = torch.cat((mask_seq, torch.zeros((batch_size,1,self.seq_embedding_size)).to(device)),1)
        mask_seq = mask_seq.to(bool)
        seq = self.seq_embedding(seq)
        #mask token for sequence
        seq_masked = seq.masked_fill(mask_seq.to(bool), 27).permute(0,2,1).to(device)

        #narrow convolution
        narrow_out_seq = self.s_narrow_conv(seq_masked)
        seq_input = narrow_out_seq.permute(0,2,1) 
        
        for block in self.block:
            mi_masked, seq_input = block(mi_masked, seq_input)
        mi_output = self.mi_dense(mi_masked)
        seq_output = self.seq_dense(seq_input)
        return mi_output, seq_output, mask
    
    def embedding(self, mi, seq):
        '''
        this function is to get the embedding of the mi before the last fully-connected layer
        '''
        device = seq.device
        batch_size = seq.shape[0]
        zero_padding = torch.zeros((batch_size,1),dtype = seq.dtype).to(device)
        seq = torch.cat((seq, zero_padding),1)
        seq = self.seq_embedding(seq).permute(0,2,1)

        #narrow convolution
        narrow_out_seq = self.s_narrow_conv(seq)
        seq_input = narrow_out_seq.permute(0,2,1) 
        count = 0 
        for block in self.block:
            if count == 2:
                mi_embedding = block.embedding(mi,seq_input)
                break
            else:        
                mi, seq_input = block(mi, seq_input)
            count += 1 
        return mi_embedding
    
    def MHA_invest(self,mi,seq):
        device = seq.device
        batch_size = seq.shape[0]
        zero_padding = torch.zeros((batch_size,1),dtype = seq.dtype).to(device)
        seq = torch.cat((seq, zero_padding),1)
        seq = self.seq_embedding(seq).permute(0,2,1)

        #narrow convolution
        narrow_out_seq = self.s_narrow_conv(seq)
        seq_input = narrow_out_seq.permute(0,2,1) 
        count = 0 
        for block in self.block:
            if count == 2:
                Att = block.MHA_invest(mi,seq_input)
                break
            else:        
                mi, seq_input = block(mi, seq_input)
            count += 1 
        return Att

    def training_step(self,batch,batch_idx):
        mi,seq = batch
        mi_re,seq_re,mask = self.forward(mi,seq) 
        loss_train_mi = self.mi_loss(mi_re[mask,:], mi[mask,:].to(torch.float16)) 
        loss_train_seq = self.seq_loss(seq_re[mask,:],seq[mask])
        train_loss = 0.5 * loss_train_mi + 0.5 * loss_train_seq 
        self.log('training loss',train_loss)
        return train_loss

    def validation_step(self,batch,batch_idx):
        mi,seq = batch
        mi_re,seq_re,mask = self.forward(mi,seq) 
        loss_valid_mi = self.mi_loss(mi_re[mask,:], mi[mask,:].to(torch.float16)) 
        loss_valid_seq = self.seq_loss(seq_re[mask,:],seq[mask])
        valid_loss = 0.5 * loss_valid_mi + 0.5 * loss_valid_seq
        self.log('valid loss',valid_loss)
        return valid_loss
    
    def test_step(self,batch):
        mi,seq = batch
        mask_mi = self.mask(seq)
        ## MI part
        #narrow convolution
        mi_re,seq_re = self.forward(mi,seq) 
        return mi_re

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 1e-5)
        return optimizer



