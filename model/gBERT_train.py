import sys
sys.path.append('/lustre/BIF/nobackup/zhang408/Lightning/iter1/')
import dataset_MI
import torch 
from Lightning.Transformer.geoBERT.geoBERT_10_13 import Whole_Module
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, Callback
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.strategies import DeepSpeedStrategy

if __name__ == "__main__":
    seed_everything(42, workers=True)
    logger = CSVLogger("CSVlog_geoBERT_10_13", name = "geoBERT_10_13")

    data = dataset_MI.proteinDM("/lustre/BIF/nobackup/zhang408/data_prepare/",
                                     4, 4)

    torch.set_float32_matmul_precision("medium")
    data.setup()
    torch.cuda.empty_cache()
    model = Whole_Module()
    
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
            deterministic=True,
            detect_anomaly=True,
            )

    trainer.fit(model,data)
    trainer.validate(model,data)
    
    torch.save(model, "/lustre/BIF/nobackup/zhang408/Lightning/Transformer/geoBERT/geoBERT_10_13_1.pt")
