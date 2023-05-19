import torchaudio

print("Downloading LibriSpeech train-clean-100 and test-clean into data directory...")
librispeech_train_clean100 = torchaudio.datasets.LIBRISPEECH("./data", "train-clean-100", download=True)
librispeech_test_clean = torchaudio.datasets.LIBRISPEECH("./data", "test-clean", download=True)
