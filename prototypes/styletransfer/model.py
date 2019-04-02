"""Model for neural style transfer."""
import tensorflow as tf
import numpy as np

from typing import Tuple

from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

from .images import process_vgg_batch


class StyleTransfer:
    def __init__(self):
        # Content layer where will pull our feature maps
        self._content_layers = ["block5_conv2"]

        # Style layer we are interested in
        self._style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]

        self._model = self._make_keras(self._style_layers, self._content_layers)

    def _make_keras(self, style_layers, content_layers) -> models.Model:
        # Returns keras model our model with access to intermediate layers.
        #
        # This function will load the VGG19 model and access the intermediate layers.
        # These layers will then be used to create a new model that will take input image
        # and return the outputs from these intermediate layers from the VGG model.
        #
        # Returns:
        #     returns a keras model that takes image inputs and outputs the style and
        #     content intermediate layers.

        # Load our model. We load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [vgg.get_layer(name).output for name in content_layers]
        model_outputs = style_outputs + content_outputs
        # Build model
        return models.Model(vgg.input, model_outputs)

    def get_content_loss(self, base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    def get_style_loss(self, base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.get_shape().as_list()
        gram_style = gram_matrix(base_style)
        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )  # / (4. * (channels ** 2) * (width * height) ** 2)

    def feature_representations(
        self, content_img: np.array, style_img: np.array
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        content_rep, style_rep = self.feature_representations_batch(
            np.expand_dims(content_img, 0), np.expand_dims(style_img, 0)
        )

        return content_rep[0], style_rep[0]

    def feature_representations_batch(
        self, content_img_batch: np.array, style_img_batch: np.array
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Helper function to compute our content and style feature representations.

        This function will simply load and preprocess both the content and style 
        images from their path. Then it will feed them through the network to obtain
        the outputs of the intermediate layers. 
        
        Arguments:
            model: The model that we are using.
            content_path: The path to the content image.
            style_path: The path to the style image
            
        Returns:
            returns the style features and the content features. 
        """

        content_img_batch = self._process_img_batch(content_img_batch)
        style_img_batch = self._process_img_batch(style_img_batch)

        style_outputs = self._model(style_img_batch)
        content_outputs = self._model(content_img_batch)

        num_style_layers = len(self._style_layers)

        # Get the style and content feature representations from our model
        style_features = [
            style_layer[0] for style_layer in style_outputs[:num_style_layers]
        ]
        content_features = [
            content_layer[0] for content_layer in content_outputs[num_style_layers:]
        ]
        return style_features, content_features

    def _process_img_batch(self, img_batch):
        # Takes a numpy image and makes it into an image processed to be ready for vgg
        return tf.convert_to_tensor(process_vgg_batch(img_batch), tf.float32)


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)

    # TODO: why do you divide by n? It could be the weighting factor
    return gram / tf.cast(n, tf.float32)

