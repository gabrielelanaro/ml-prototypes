import numpy as np
from prototypes.styletransfer.images import load_image

TEST_IMG_PATH = "prototypes/styletransfer/tests/img_lights.jpg"


def test_load_image():

    data = load_image(TEST_IMG_PATH)

    assert isinstance(data, np.ndarray)
    assert data.shape == (341, 512, 3)

