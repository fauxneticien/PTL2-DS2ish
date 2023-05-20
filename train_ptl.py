import argparse
import torch
import lightning.pytorch as pl

from datamodules.libris import LibrisDataModule
from datamodules._utils import TextTransform, MelSpecWithTrainSpecAug, GreedyDecoder
from models.DeepSpeech2ish import DeepSpeech2ishModel, DeepSpeech2ishLightningModule

parser = argparse.ArgumentParser(description='Train a DeepSpeech2ish model using PyTorch Lightning.')
parser.add_argument('--random_seed', type=int, default=3)
args = parser.parse_args()

pl.seed_everything(args.random_seed)

wandb_logger = pl.loggers.WandbLogger(
    project='PLT2-DeepSpeech2ish',
    name=f"PTL, seed={args.random_seed}"
)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

trainer = pl.trainer.trainer.Trainer(
    max_epochs=2,
    accelerator="gpu",
    devices=1,
    check_val_every_n_epoch=1,
    log_every_n_steps=100,
    logger=wandb_logger,
    callbacks=[lr_monitor],
    enable_checkpointing=False
)

data_module = LibrisDataModule(
    dataset_path="./data",
    batch_size=10,
    random_seed=args.random_seed,
    audio_transforms=MelSpecWithTrainSpecAug(),
    text_transform=TextTransform()
)

model_module = DeepSpeech2ishLightningModule(
    model=DeepSpeech2ishModel(),
    eval_decoder=GreedyDecoder(data_module.text_transform)
)

trainer.fit(model_module, datamodule=data_module)
