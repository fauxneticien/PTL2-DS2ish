# PTL2-DS2ish

This repository takes [AssemblyAI's end-to-end ASR tutorial](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/) by [Michael Nguyen](https://www.assemblyai.com/blog/author/michael/) as a starting point and converts the training code to be compatible with the latest PyTorch Lightning release (2.0.2 as of May 2023), mainly relying on the official PyTorch Lightning [preparation guide](https://lightning.ai/docs/pytorch/stable/starter/converting.html) and the [DNN Beamformer example](https://github.com/pytorch/audio/tree/main/examples/dnn_beamformer) by [
Zhaoheng Ni](https://nateanl.github.io/) in torchaudio as a template.

## Motivation

The motivation for this repository is to have an easily-understandable test harness/template for future ASR experiments, hence the choice to use a relatively small/simple model such as the (adapted) Deep Speech 2 implementation found in the AssemblyAI tutorial as a starting point.
