import sys
import torch
sys.path.append('/geoBERT/')
from geoBERT_10_5 import Whole_Module
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, Callback
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from lightning.pytorch import seed_everything
sys.path.append('/lustre/BIF/nobackup/zhang408/Lightning/iter1/')
import dataset_TF

if __name__ == "__main__":
    seed_everything(42, workers=True)
    logger = CSVLogger("CSVlog_geoBERT_10_5", name = "geoBERT_10_5")

    data = dataset_TF.proteinDM("/lustre/BIF/nobackup/zhang408/data_prepare/",
                                    "/lustre/BIF/nobackup/zhang408/cafa5/Train/",
                                     16, 4)

    torch.set_float32_matmul_precision("medium")
    data.setup()
    torch.cuda.empty_cache()
    model = Whole_Module()
    ckpt = '/lustre/BIF/nobackup/zhang408/Lightning/Transformer/geoBERT/CSVlog_geoBERT_10_5/geoBERT_10_5/version_17/checkpoints/epoch=19-step=66260.ckpt'
    #return NaN loss, change learning rate from 2e-5 to 1e-5
    model = model.load_from_checkpoint(ckpt)
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

    torch.save(model, "/lustre/BIF/nobackup/zhang408/Lightning/Transformer/geoBERT/geoBERT_10_5_1.pt")
