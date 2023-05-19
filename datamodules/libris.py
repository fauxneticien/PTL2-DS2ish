import lightning.pytorch as pl

from torch import Tensor
from typing import Dict, Tuple, Union, List
from ._utils import TextTransform

import torch
import torch.nn as nn
import torchaudio

text_transform = TextTransform()

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

def data_processing(data, data_type="train"):

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

class LibrisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "train-clean-100")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: data_processing(x, 'train'),
            shuffle=False,
            num_workers=8,
        )

    def val_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "test-clean")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: data_processing(x, 'valid'),
            shuffle=False,
            num_workers=8,
        )
