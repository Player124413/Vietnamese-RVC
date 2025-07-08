import torch

PITCH_BINS = 1440

class MODEL(torch.nn.Sequential):
    def __init__(self):
        layers = (Block(1, 256, 481, (2, 2)), Block(256, 32, 225, (2, 2)), Block(32, 32, 97, (2, 2)), Block(32, 128, 66), Block(128, 256, 35), Block(256, 512, 4), torch.nn.Conv1d(512, PITCH_BINS, 4))
        super().__init__(*layers)

    def forward(self, frames):
        return super().forward(frames[:, :, 16:-15])

class Block(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, length=1, pooling=None, kernel_size=32):
        layers = (torch.nn.Conv1d(in_channels, out_channels, kernel_size), torch.nn.ReLU())

        if pooling is not None: layers += (torch.nn.MaxPool1d(*pooling),)
        layers += (torch.nn.LayerNorm((out_channels, length)),)

        super().__init__(*layers)