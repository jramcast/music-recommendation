from pathlib import Path
import torch
from cnn import CNNNetwork
from torch import nn
import torchaudio
from datasets.fourq import FourQDataset


def predict(model: nn.Module, input: torch.Tensor, target_index: int):
    model.eval()
    with torch.no_grad():
        predictions: torch.Tensor = model(input)
        # Tensor (1, 4) -> [ [0.2, 0.4, 0.05, 0.3] ]
        predicted_index = predictions[0].argmax(0)
        predicted_class = FourQDataset.get_quadrant_from_index(int(predicted_index))
        expected_class = FourQDataset.get_quadrant_from_index(target_index)

    return predicted_class, expected_class


if __name__ == "__main__":
    model = CNNNetwork()
    model_weights = torch.load("cnn.pth")
    model.load_state_dict(model_weights)

    # Load dataset
    AUDIO_DIR = Path(__file__).parent.joinpath("../../data/4q/audio/").absolute()
    ANNOTATIONS_FILE = AUDIO_DIR.joinpath("panda_dataset_taffc_annotations.csv")
    SAMPLE_RATE = 22050
    SAMPLE_SECONDS = 30
    NUM_SAMPLES = SAMPLE_SECONDS * SAMPLE_RATE
    FRAME_SIZE = 1024
    HOP_LENGTH = int(FRAME_SIZE / 2)

    # Training parameters
    BATCH_SIZE = 512
    EPOCHS = 10
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

    hits = 0
    for i in range(len(dataset)):
        sample_index = i
        # sample_input is Tensor [num_channels, fr, time]
        sample_input = dataset.get_input_from_sample(sample_index)
        sample_class_index = dataset.get_quadrant_from_sample(sample_index)

        # Model expect the extra batch dimension: [batch_size, num_channels, fr, time]
        # 0 is the index where we introduce the new dimension
        sample_input = sample_input.unsqueeze(0)
        predicted, expected = predict(model, sample_input, sample_class_index)

        if predicted == expected:
            hits += 1

        print(f"Predicted: {predicted}. Expected: {expected}")

    print("Accuracy: ", hits / len(dataset))
