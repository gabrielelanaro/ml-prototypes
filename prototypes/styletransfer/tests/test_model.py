import numpy as np
import tensorflow as tf
from prototypes.styletransfer.model import StyleTransfer, gram_matrix
import tensorflow.contrib.eager as tfe


rng = np.random.RandomState(42)


def _sample_img(size):
    return rng.randint(0, 255, (size, size, 3)).astype("float32")


def test_model():
    content_img = _sample_img(512)
    style_img = _sample_img(512)

    model = StyleTransfer()

    style_rep, content_rep = model.feature_representations(content_img, style_img)
    assert isinstance(content_rep, list)
    assert isinstance(style_rep, list)


def test_model_loss():
    # NOTE: you're not supposed to test private methods but those are
    # complicated private methods that justify their usage. We could
    # also refactor to make more things public and testable.
    content_img = _sample_img(512)
    style_img = _sample_img(512)
    init_img = _sample_img(512)

    model = StyleTransfer()

    content_rep, style_rep = model.feature_representations(content_img, style_img)

    init_img = model._process_img_batch(init_img)
    init_img = tfe.Variable(init_img, dtype=tf.float32)

    gram_style_features = [gram_matrix(style_feature) for style_feature in style_rep]
    loss_weights = (0.5, 0.5)
    losses = model._compute_loss(
        loss_weights, init_img, gram_style_features, content_rep
    )

    assert isinstance(losses, tuple)
    assert isinstance(losses[0], tf.Tensor)


def test_run_styletransfer():
    content_img = _sample_img(512)
    style_img = _sample_img(512)

    model = StyleTransfer()
    content_rep, style_rep = model.feature_representations(content_img, style_img)

    for st in model.run_style_transfer(content_img, style_img, num_iterations=10):
        assert isinstance(st.image, np.ndarray)
