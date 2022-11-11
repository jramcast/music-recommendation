from pathlib import Path
from typing import Union
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class FourQDataset(Dataset):
    def __init__(
        self,
        annotations_file: Union[Path, str],
        audio_dir: Union[Path, str],
        transformation: torch.nn.Module,
        sample_rate: int,
        num_samples: int,
        device: str,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transformation.to(device)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):
        """
        Returns (Tensor, quadrant) tuple
        The tensor returns the result of applying the transformation
        to the mono audio file:
        Example: (1, bins/mels, frames)
        """
        song_audio_path = self._get_song_audio_path(index)
        quadrant = self._get_song_quadrant(index)

        signal = self._load_audio_signal(song_audio_path)
        return signal, quadrant

    def _load_audio_signal(self, song_audio_path) -> torch.Tensor:
        signal, sr = torchaudio.load(song_audio_path)  # type: ignore
        signal = signal.to(self.device)
        signal = self._resample_if_required(signal, sr)
        signal = self._convert_to_mono_if_stereo(signal)
        signal = self._cut_if_required(signal)
        signal = self._right_pad_if_required(signal)
        signal = self.transform(signal)
        return signal

    def _resample_if_required(self, signal: torch.Tensor, original_sr: int):
        if original_sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(original_sr, self.sample_rate)
            signal = resample(signal)

        return signal

    def _convert_to_mono_if_stereo(self, signal: torch.Tensor):
        """Expects tensor (num_channels, samples) and returns (1, samples)"""
        channels_dim = 0

        if self._signal_is_stereo(signal):
            # If keepdim is True, then the output tensor is of the same size as
            # input except in the dimension(s) dim where it is of size 1.
            signal = torch.mean(signal, dim=channels_dim, keepdim=True)

        return signal

    def _cut_if_required(self, signal: torch.Tensor):
        """
        Expects tensor (num_channels, samples)
        and returns (num_channels, self.num_sampes)
        """
        samples_dim = 1
        num_samples = signal.shape[samples_dim]
        if num_samples > self.num_samples:
            signal = signal[:, : self.num_samples]

        return signal

    def _right_pad_if_required(self, signal: torch.Tensor):
        samples_dim = 1
        num_samples = signal.shape[samples_dim]

        if num_samples < self.num_samples:
            left_padding = 0
            right_padding = self.num_samples - num_samples

            # This only contain padding values for the last dimension
            padding = (left_padding, right_padding)
            signal = torch.nn.functional.pad(signal, padding)  # type: ignore

        return signal

    def _signal_is_stereo(self, signal):
        channels_dim = 0
        return signal.shape[channels_dim] > 1

    def _get_song_audio_path(self, index):
        song_id = self._get_audio_sample_id(index)
        quadrant = self._get_song_quadrant(index)

        path = Path(self.audio_dir).joinpath(quadrant).joinpath(f"{song_id}.mp3")
        return str(path)

    def _get_audio_sample_id(self, index):
        """
        For an index, get the song ID.
        For example:
        0 -> MT0000004637
        1 -> MT0000011357
        """
        song_id_column = 0
        song_id = self.annotations.iloc[index, song_id_column]
        return song_id

    def _get_song_quadrant(self, index):
        """
        Return Q1, Q2, Q3, or Q4, based on the song index
        """
        quadrant_column = 1
        quadrant = self.annotations.iloc[index, quadrant_column]
        return str(quadrant)


if __name__ == "__main__":
    AUDIO_DIR = Path("data/4q/audio/").absolute()
    ANNOTATIONS_FILE = "data/4q/audio/panda_dataset_taffc_annotations.csv"
    SAMPLE_RATE = 22050
    SAMPLE_SECONDS = 30
    NUM_SAMPLES = SAMPLE_SECONDS * SAMPLE_RATE
    FRAME_SIZE = 1024
    HOP_LENGTH = int(FRAME_SIZE / 2)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using device: ", device)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=64
    )

    dataset = FourQDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mel_spectrogram_transform,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
    )

    print("Dataset length: ", len(dataset))

    signal, quadrant = dataset[0]
    print("Sample 0 signal shape (channels, mels, frames)", signal.size())
    print("Sample 0 signal number of samples", signal.size()[2] * HOP_LENGTH)
    print(
        "Sample 0 signal number of seconds", signal.size()[2] * HOP_LENGTH / SAMPLE_RATE
    )
    print("Sample 0 quadrant", quadrant)
