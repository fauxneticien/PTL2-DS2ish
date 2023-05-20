import lightning.pytorch as pl

from torch import Tensor
from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torchaudio

class LibrisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        random_seed: int,
        text_transform,
        audio_transforms
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.random_seed=random_seed
        self.audio_transforms=audio_transforms
        self.text_transform=text_transform

    def _collate_fn(self, data, data_type="train"):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (waveform, _, utterance, _, _, _) in data:
            if data_type == 'train':
                spec = self.audio_transforms.train_transform(waveform).squeeze(0).transpose(0, 1)
            elif data_type == 'valid':
                spec = self.audio_transforms.valid_transform(waveform).squeeze(0).transpose(0, 1)
            else:
                raise Exception('data_type should be train or valid')
            spectrograms.append(spec)
            label = torch.Tensor(self.text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0]//2)
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths
    
    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.random_seed)

        dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "train-clean-100")

        # For debugging, do manual shuffle of training data:
        import numpy as np
        indices=np.arange(len(dataset))
        rng = np.random.default_rng(seed=self.random_seed)
        rng.shuffle(indices)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Train indices: {indices}")

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self._collate_fn(x, 'train'),
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            generator=g
        )

    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.random_seed)

        dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "test-clean")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self._collate_fn(x, 'valid'),
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            generator=g
        )
