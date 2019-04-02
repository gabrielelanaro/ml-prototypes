import numpy as np
import tensorflow as tf
from prototypes.styletransfer.model import StyleTransfer

rng = np.random.RandomState(42)


def _sample_img(size):
    return rng.randint(0, 255, (size, size, 3)).astype("float32")


def test_model():
    content_img = _sample_img(512)
    style_img = _sample_img(512)

    model = StyleTransfer()

    content_rep, style_rep = model.feature_representations(content_img, style_img)

    assert isinstance(content_rep, tf.Tensor)
    assert isinstance(style_rep, tf.Tensor)
