import lightning.pytorch as pl
import torch
import torchaudio
import jiwer

class DeepSpeech(torch.nn.Module):
    def __init__(self, n_mels: int = 80, n_class: int = 29):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)
        self.model = torchaudio.models.DeepSpeech(n_feature=n_mels, n_class=n_class)

    def forward(self, batch):
        input_feats = self.mel_spec(batch).transpose(2, 1)
        outputs = self.model(input_feats).transpose(0,1)
        return outputs
    
class DeepSpeechLightningModule(pl.LightningModule):

    def __init__(self, model: torch.nn.Module, id2label: dict):
        super(DeepSpeechLightningModule, self).__init__()
        self.model = model
        self.ctc_loss = torch.nn.CTCLoss(blank=0)

        self.id2label = id2label

        self.outs = {"refs" : [], "hyps": []}

    def training_step(self, batch, batch_idx):
        data, labels, audio_lengths, label_lengths = batch
        feat_lengths = torch.ceil(audio_lengths / 200).int()
        outputs = self.model(data)
        loss=self.ctc_loss(outputs, labels, feat_lengths, label_lengths)
        self.log("train/loss", loss.item(), sync_dist=True)
        return loss

    def int2label(self, ints):
        return ("".join([ self.id2label[i] for i in ints.to('cpu').numpy() if i > 0 ]).replace("|", " ")).strip()

    def validation_step(self, batch, batch_idx):
        data, labels, audio_lengths, label_lengths = batch
        feat_lengths = torch.ceil(audio_lengths / 200).int()
        outputs = self.model(data)
        loss=self.ctc_loss(outputs, labels, feat_lengths, label_lengths)
        
        if not self.trainer.sanity_checking:
            self.outs["hyps"].extend([ self.int2label(a) for a in outputs.transpose(0,1).argmax(dim=2) ])
            self.outs["refs"].extend([ self.int2label(l) for l in labels ])

        self.log("val/loss", loss.item(), sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self, *_):
        if not self.trainer.sanity_checking:
            wer = jiwer.wer(self.outs["refs"], self.outs["hyps"])
            self.log("val/wer", wer, sync_dist=True)
            self.outs = {"refs" : [], "hyps": []}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
        }
