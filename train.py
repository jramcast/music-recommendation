from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.fourq import FourQDataset
from cnn import CNNNetwork


def create_data_loader(train_data: Dataset, batch_size: int):
    return DataLoader(train_data, batch_size=batch_size)


def train(
    model: nn.Module, loader: DataLoader, loss_fn, optimiser, device: str, epochs: int
):
    for i in range(epochs):
        print("Epoch", i + 1)
        train_one_epoch(model, loader, loss_fn, optimiser, device)
        print("-" * 30)

    print("Finished training")


def train_one_epoch(
    model: nn.Module, loader: DataLoader, loss_fn, optimiser, device: str
):
    loss = None

    for inputs, targets in loader:
        print("input", inputs.shape)
        # Send data to GPU if available
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Predict and calculate loss
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)

        # Backpropagation
        optimiser.zero_grad()  # reset gradients for this iteration
        loss.backward()  # run backprop
        optimiser.step()  # update the weights

    if loss:
        print("loss:", loss.item())


if __name__ == "__main__":

    # Dataset parameters
    AUDIO_DIR = Path("data/4q/audio/").absolute()
    ANNOTATIONS_FILE = AUDIO_DIR.joinpath("panda_dataset_taffc_annotations.csv")
    SAMPLE_RATE = 22050
    SAMPLE_SECONDS = 30
    NUM_SAMPLES = SAMPLE_SECONDS * SAMPLE_RATE
    FRAME_SIZE = 1024
    HOP_LENGTH = int(FRAME_SIZE / 2)

    # Training parameters
    BATCH_SIZE = 512
    EPOCHS = 1
    LEARNING_RATE = 0.001

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using device: ", device)

    # Get dataset
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

    train_data_loader = create_data_loader(dataset, BATCH_SIZE)

    # Create model
    model = CNNNetwork().to(device)
    print(model)

    # Create loss funct and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    train(model, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    # Save model
    model_path = "cnn.pth"
    torch.save(model.state_dict(), model_path)
    print("Model saved at ", model_path)
