import lightning.pytorch as pl

from datamodules.libris import LibrisDataModule, _get_id2label
from model import DeepSpeech, DeepSpeechLightningModule

pl.seed_everything(1)

trainer = pl.trainer.trainer.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    check_val_every_n_epoch=10,
    log_every_n_steps=1,
    logger=None,
)

model = DeepSpeech()
id2label = _get_id2label()

model_module = DeepSpeechLightningModule(model, id2label)

data_module = LibrisDataModule(dataset_path="./data", batch_size=1)

trainer.fit(model_module, datamodule=data_module)
