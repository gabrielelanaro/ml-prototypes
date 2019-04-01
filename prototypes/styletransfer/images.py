import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image


def load_image(img_path: str, max_size: int = 512) -> np.array:
    """Load an image from a file.

    img_path: The path of the image
    max_size: The maximum size of the image

    The returned image is represented as an array of floating point
    numbers between 0 and 255.
    """
    img = Image.open(img_path)
    longest_size = max(img.size)
    scale = max_size / longest_size

    new_width = round(img.size[0] * scale)
    new_height = round(img.size[1] * scale)

    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    return img


def show_image(img_array: np.array, title: str = None) -> None:
    out = out.astype("uint8")
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def process_vgg(img_array: np.array):
    return tf.keras.applications.vgg19.preprocess_input(img_array)


def deprocess_vgg(img_array: np.array):
    x = img_array.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, (
        "Input to deprocess image must be an image of "
        "dimension [1, height, width, channel] or [height, width, channel]"
    )
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype("uint8")
    return x
