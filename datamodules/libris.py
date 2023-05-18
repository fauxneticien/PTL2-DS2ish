import lightning.pytorch as pl

from torch import Tensor
from typing import Dict, Tuple, Union, List

import torch
import torchaudio

class CollateFnLibriLightLimited:
    """The collate class for LibriSpeech or LibriLightLimited dataset."""

    def __call__(self, batch: List[Tuple[Tensor, int, str, int, int, int]]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch (List(Tuple(Tensor, int, str, int, int, int))):
                The list of tuples that contains
                waveform, sample_rate, transcript, speaker_id, chapter_id, and utterance_id.

        Returns:
            (Tuple(Tensor, Tensor, Tensor, Tensor)):
                The Tensor of waveforms with dimensions `(batch, time)`.
                The Tensor of labels with dimensions `(batch, seq)`.
                The Tensor of audio lengths with dimensions `(batch,)`.
                The Tensor of length lengths with dimensions `(batch,)`.

        """
        audio_sizes = [sample[0].shape[1] for sample in batch]
        audio_size = max(audio_sizes)
        waveforms, labels, audio_lengths, label_lengths = [], [], [], []
        label2id = _get_label2id()
        for sample in batch:
            waveform, transcript = sample[0], sample[2]
            # add one "|" symbol after the end of transcription as the word termination
            transcript = transcript + "|"
            label = torch.tensor([label2id[e] for e in transcript.replace(" ", "|").upper()])
            audio_length = waveform.size(1)
            label_length = label.size(0)
            waveforms.append(waveform)
            audio_lengths.append(audio_length)
            label_lengths.append(label_length)
            labels.append(label)

        data = torch.zeros(len(batch), audio_size)
        for i in range(len(waveforms)):
            data[i][0 : waveforms[i].shape[1]] = waveforms[i]
        audio_lengths = torch.tensor(audio_lengths)
        label_lengths = torch.tensor(label_lengths)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
        return data, labels.int(), audio_lengths.int(), label_lengths.int()

def _get_id2label() -> Dict:
    """Get the dictionary that maps indices of ASR model's last layer dimension to the corresponding labels."""
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    labels = bundle.get_labels()
    return {i: char.lower() for i, char in enumerate(labels)}

def _get_label2id() -> Dict:
    """Get the dictionary that maps the labels to the corresponding indices in ASR model's last dimension."""
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    labels = bundle.get_labels()
    return {char: i for i, char in enumerate(labels)}

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
        dataset = torchaudio.datasets.LibriLightLimited(self.dataset_path, subset="10min")

        # Try to over-fit to single data point
        dataset = torch.utils.data.Subset(dataset, [0])
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=CollateFnLibriLightLimited(),
            shuffle=False,
            num_workers=8,
        )

    def val_dataloader(self):
        # dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "dev-other")
        # dataset = torch.utils.data.Subset(dataset, list(range(0, len(dataset), 2)))

        # Try to over-fit to single data point
        dataset = torchaudio.datasets.LibriLightLimited(self.dataset_path, subset="10min")
        dataset = torch.utils.data.Subset(dataset, [0])
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=CollateFnLibriLightLimited(),
            shuffle=False,
            num_workers=8,
        )
