# PTL2-DS2ish

This repository takes [AssemblyAI's end-to-end ASR tutorial](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/) by [Michael Nguyen](https://www.assemblyai.com/blog/author/michael/) as a starting point and converts the training code to be compatible with the latest PyTorch Lightning release (2.0.2 as of May 2023), mainly relying on the official PyTorch Lightning [preparation guide](https://lightning.ai/docs/pytorch/stable/starter/converting.html) and the [DNN Beamformer example](https://github.com/pytorch/audio/tree/main/examples/dnn_beamformer) by [
Zhaoheng Ni](https://nateanl.github.io/) in torchaudio as a template.

## Motivation

The motivation for this repository is to have an easily-understandable test harness/template for future ASR experiments, hence the choice to use a relatively small/simple model such as the (adapted) Deep Speech 2 implementation found in the AssemblyAI tutorial as a starting point.

## Roadmap

### May 21, 2023

- Verified that adapted code in PyTorch Lightning behaves as original pure PyTorch code (reduced `max_epochs` in runs 2-5 down to 30 for faster training time).

    <img width="348" alt="Screenshot 2023-05-21 at 5 41 10 PM" src="https://user-images.githubusercontent.com/9938298/239777087-7713857e-81ae-4dc2-b9a1-eb25497a3ac9.png">

    | seed  | epochs | WER (PyTorch Lightning code) | WER (AssemblyAI code)
    | ------------- | ------------- | ------------- | ------------- |
    | 1  | 100  | 0.2897 | 0.2922 |
    | 2  | 30  | 0.3373 | 0.3421 |
    | 3  | 30  | 0.3431 | 0.3416 |
    | 4  | 30  | 0.3408 | 0.3445 |
    | 5  | 30  | 0.3464 | 0.3452 |
