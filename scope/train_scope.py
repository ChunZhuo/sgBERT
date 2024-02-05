import sys
sys.path.append('/lustre/BIF/nobackup/zhang408/Lightning/Transformer/')
from Basic_model import scope
from plot_data import proteinDM
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
import seaborn as sns

def MHA_invest(model):

    data = proteinDM("/lustre/BIF/nobackup/zhang408/data_prepare/scope/data_10000/", 1, 4)
    #data.setup()
    data_ready = data.subdata("sample_MI_10000","pdb_classes_10000","samples_10000.fasta")
    prot = list(data_ready.keys())[7]
    print(prot)
    print(len(data_ready[prot][1]))
    
    input = (torch.tensor(data_ready[prot][0]).unsqueeze(0), 
             torch.tensor(data_ready[prot][1]).unsqueeze(0),
             int(data_ready[prot][2]))
    model.eval()
    Att = model.MHA_invest(input)

    return Att 
    
if __name__ == "__main__":

    logger = CSVLogger("CSVlog_Basic_model_scope", name = "geometricus")
    model = scope(4,4)
    data = proteinDM("/lustre/BIF/nobackup/zhang408/data_prepare/scope/data_10000/", 4, 4)
    data.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
            strategy='ddp_find_unused_parameters_true',
            logger = logger,
            accelerator = 'gpu',
            devices = 2,
            min_epochs =2,
            max_epochs = 10000,
            precision = "16-mixed",
            callbacks = EarlyStopping(monitor="valid loss"),
            )

    trainer.fit(model,data)
    trainer.validate(model,data)


    '''
    #Attention head visualization
    model = scope(4,4,True,False)
    Att = MHA_invest(model)
    ckpt = torch.load('/lustre/BIF/nobackup/zhang408/Lightning/Transformer/scope_prediction/CSVlog_sgBERT_scope_frozen/geoBERT_10_5_1/version_1/checkpoints/epoch=13-step=26362.ckpt')
    model1 = scope(4,4,True,False)
    model1.load_state_dict(ckpt['state_dict'])
    Att_ft = MHA_invest(model1)
    
    Att = Att.squeeze().detach().sum(-2)
    Att_ft = Att_ft.squeeze().detach().sum(-2)
    Att_norm = Att/(torch.max(Att)- torch.min(Att))
    Att_ft_norm = Att_ft/(torch.max(Att_ft) - torch.min(Att_ft))
    print(Att_norm.shape)
    print(Att_ft_norm.shape)
    
    heads = sns.heatmap(Att_norm,linewidths = 0.5, cmap="YlGnBu")
    fig = heads.get_figure()
    fig.savefig('sgBERT_heads_norm_4.png')
    
    heads_ft = sns.heatmap(Att_ft_norm, linewidths = 0.5, cmap ="YlGnBu")
    fig1 = heads_ft.get_figure()
    fig1.savefig('sgBERT_heads_ft_norm_3.png')
    '''