from typing import Any
import torch
import torchaudio

class TextTransform:
    # Adapted from https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/

    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

class MelSpecWithTrainSpecAug:

    def __init__(self, n_mels=128, sample_rate=16_000, freq_mask_param=30, time_mask_param=100):
        self.train_audio_transforms = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        )

        self.valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

    def train_transform(self, waveform):
        return self.train_audio_transforms(waveform)
    
    def valid_transform(self, waveform):
        return self.valid_audio_transforms(waveform)

class GreedyDecoder:

    def __init__(self, text_transform):
        self.text_transform = text_transform

    def __call__(self, output, labels, label_lengths, blank_label=28, collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(self.text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j -1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.text_transform.int_to_text(decode))
        return decodes, targets
