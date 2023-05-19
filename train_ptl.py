import lightning.pytorch as pl

from datamodules.libris import LibrisDataModule
from models.DeepSpeech2ish import DeepSpeech2ishModel, DeepSpeech2ishLightningModule

pl.seed_everything(7)

trainer = pl.trainer.trainer.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    check_val_every_n_epoch=10,
    log_every_n_steps=1,
    logger=None,
)

model = DeepSpeech2ishModel()

model_module = DeepSpeech2ishLightningModule(model)

data_module = LibrisDataModule(dataset_path="./data", batch_size=10)

trainer.fit(model_module, datamodule=data_module)
