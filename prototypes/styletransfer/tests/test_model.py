import numpy as np

rng = np.random.RandomState(42)


def _sample_img(size):
    return rng.randint(0, 255, (size, size, 3), np.float)


def test_model():
    content_img = _sample_img(512)
    style_img = _sample_img(512)

    model = StyleTransfer()

    content_rep, style_rep = model.feature_representations(content_img, style_img)
