import argparse
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from datamodules.libris import LibrisDataModule
from datamodules._utils import TextTransform, MelSpecWithTrainSpecAug, GreedyDecoder
from models.DeepSpeech2ish import DeepSpeech2ishModel, DeepSpeech2ishLightningModule

class DeepSpeech2ishLightningModuleForAMP(DeepSpeech2ishLightningModule):

    def __init__(self, model: torch.nn.Module, eval_decoder):
        super().__init__(model, eval_decoder)
        # To check: is the motivation for specifying reduction="sum", zero_infinity=True because of strategy="ddp"?
        self.ctc_loss = torch.nn.CTCLoss(blank=28, reduction="sum", zero_infinity=True)
        self.scaler = torch.cuda.amp.GradScaler()

    def training_step(self, batch, batch_idx):
        # Adapted from HuBERT fine-tuning recipe using PyTorch Lightning by Zhaoheng Ni
        # 
        # https://github.com/pytorch/audio/blob/8a893fb3bbcb4b9707c7496b5e7547a0a6b2e288/examples/hubert/lightning.py#L462
        spectrograms, labels, input_lengths, label_lengths = batch 
        
        opt = self.optimizers()
        opt.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.model(spectrograms)
            outputs = F.log_softmax(outputs, dim=2)
            outputs = outputs.transpose(0, 1) # (time, batch, n_class)
    
            loss = self.ctc_loss(outputs, labels, input_lengths, label_lengths)

        # normalize the loss based on the sum of batch_sie across all GPUs
        batch_size = batch[0].size(0)
        batch_sizes = self.all_gather(batch_size)
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size

        # backward the loss and clip the gradients
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

        # optimization
        self.scaler.step(opt)
        sch = self.lr_schedulers()
        sch.step()
        self.scaler.update()

        self.log("train/loss", loss.item(), prog_bar=True, sync_dist=True)

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
    max_epochs=30,
    accelerator="gpu",
    devices=1,
    check_val_every_n_epoch=1,
    log_every_n_steps=100,
    logger=wandb_logger,
    callbacks=[lr_monitor],
    enable_checkpointing=False,
    # Hard-code strategy to ddp for use with custom training_step() even if using just 1 GPU
    strategy="ddp"
)

data_module = LibrisDataModule(
    dataset_path="./data",
    batch_size=10,
    random_seed=args.random_seed,
    audio_transforms=MelSpecWithTrainSpecAug(),
    text_transform=TextTransform()
)

model_module = DeepSpeech2ishLightningModuleForAMP(
    model=DeepSpeech2ishModel(),
    eval_decoder=GreedyDecoder(data_module.text_transform)
)

trainer.fit(model_module, datamodule=data_module)
