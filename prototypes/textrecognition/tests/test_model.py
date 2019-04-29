import numpy as np
import torch
from prototypes.textrecognition.model import ConvolutionalEncoder, TextRecognizer
from prototypes.textrecognition.trainer import Trainer

rng = np.random.RandomState(42)

batch_size = 2
channels = 3  # RGB
width = 100
height = 32
_test_img = rng.rand(batch_size, channels, height, width)
_test_tensor = torch.Tensor(_test_img)

seq_len = 7
input_features = 128
_test_seq = rng.rand(seq_len, batch_size, input_features)


def test_cnn():

    enc = ConvolutionalEncoder(3)
    out_batch, out_channels, out_height, out_width = enc(_test_tensor).shape

    assert enc.output_size(batch_size, 3, height, width) == (
        out_batch,
        out_channels,
        out_height,
        out_width,
    )

    assert out_batch == batch_size
    assert out_channels == 512
    assert out_height == 1
    assert (
        out_width == 24
    )  # TODO(glanaro): paper says it shoul be 25, I guess close enough


def test_rnn():
    rnn = torch.nn.LSTM(
        input_size=input_features, hidden_size=16, num_layers=2, bidirectional=True
    )
    output, _ = rnn(torch.Tensor(_test_seq))
    out_seq_len, out_batch_size, out_features = output.shape

    assert out_batch_size == batch_size
    assert out_seq_len == seq_len
    assert out_features == 16 * 2


def test_recognizer():

    rec = TextRecognizer(vocabulary_size=4)
    output = rec(_test_tensor)
    out_seq_len, out_batch_len, out_vocab_size = output.shape

    assert out_vocab_size == 4
    assert out_batch_len == batch_size
    assert out_seq_len == 24  # This depends on the input size


def test_training():

    trainer = Trainer(100)

    trainer.train(_test_img, ["abc", "bcd"])
