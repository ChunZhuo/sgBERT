import torch 
from Basic_model import scope 
from torchmetrics.classification import MulticlassAccuracy 
import pytorch_lightning as pl 
import sys
sys.path.append('/lustre/BIF/nobackup/zhang408/Lightning/Transformer/')
from plot_data import proteinDM

model = scope.\
    load_from_checkpoint(\
        '/lustre/BIF/nobackup/zhang408/Lightning/Transformer/scope_prediction/CSVlog_Basic_model_scope/geometricus/version_0/checkpoints/epoch=94-step=178885.ckpt',
            num_feature= 4,
            num_class = 4,
            #add_seq = True,
            #freeze = True
            ).to('cpu')

data = proteinDM("/lustre/BIF/nobackup/zhang408/data_prepare/scope/data_10000/", 16, 4)
data.setup()
valid_set = data.val_dataloader()
y_hat = torch.tensor([])
y = torch.tensor([]) 

for batch in valid_set:
    output = model.test_step(batch)
    y_hat = torch.cat((y_hat, output),dim = 0)
    #print(batch[2])
    y = torch.cat((y,batch[2]),dim=0)

metric = MulticlassAccuracy(num_classes=4)
accuracy = metric(y_hat, y)
print(accuracy)

