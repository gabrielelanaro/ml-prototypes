from torch import nn
from .convolutional import ConvolutionalEncoder


class MapToSequence(nn.Module):
    def forward(self, x):
        batch_shape, channels, height, width = x.shape
        assert height == 1  # For now, otherwise we need to concatenate
        x = x.squeeze(dim=2)
        return x.permute(2, 0, 1)


class TextRecognizer(nn.Module):
    def __init__(
        self,
        n_channels=3,
        sequence_hidden_size=32,
        sequence_num_layers=2,
        vocabulary_size=4,
    ):
        super().__init__()
        self.image_encoder = ConvolutionalEncoder(n_channels)
        self.maptoseq = MapToSequence()
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=sequence_hidden_size,
            num_layers=sequence_num_layers,
            bidirectional=True,
        )
        self.linear_to_vocab = nn.Linear(
            sequence_hidden_size * sequence_num_layers, vocabulary_size
        )

    def forward(self, img_data):
        feature_maps = self.image_encoder(img_data)
        seq = self.maptoseq(feature_maps)
        out, _ = self.rnn(seq)
        logits = self.linear_to_vocab(out)
        return logits

    def sequence_size(self, width):
        return self.image_encoder.output_size(1, 3, 32, width)[3]
