"""Model for neural style transfer."""
import tensorflow as tf
import numpy as np
import time
import enum
import scipy

from typing import NamedTuple, Tuple, List, Iterator

import tensorflow.contrib.eager as tfe

from tensorflow.python.keras import models

from .images import process_vgg, deprocess_vgg

tf.enable_eager_execution()


class InitType(enum.Enum):
    RANDOM = "random"
    CONTENT = "content"


class StyleTransferResult(NamedTuple):
    image: np.array
    iteration_no: int
    total_loss: float
    style_loss: float
    content_loss: float
    elapsed_time_sec: float


class StyleTransfer:
    def __init__(self, init_image_type: InitType = InitType.RANDOM):
        """Applies style transfer to an image.

        Parameters:
        - init_image_type: specifies how to initialize the style transfer.
          Either using the original content image, or at random.
        """
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
        self._num_style_layers = len(self._style_layers)
        self._num_content_layers = len(self._content_layers)

        self._model = self._make_keras(self._style_layers, self._content_layers)

        # Configurable option
        self.init_image_type = init_image_type

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

    def feature_representations(
        self, content_img: np.array, style_img: np.array
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

        content_img = self._process_img(content_img)
        style_img = self._process_img(style_img)

        style_outputs = self._model(style_img)
        content_outputs = self._model(content_img)

        self._num_style_layers = len(self._style_layers)

        # Get the style and content feature representations from our model
        style_features = [
            style_layer[0] for style_layer in style_outputs[: self._num_style_layers]
        ]
        content_features = [
            content_layer[0]
            for content_layer in content_outputs[self._num_style_layers :]
        ]
        return content_features, style_features

    def _estimate_content2weight(
        self,
        content_img: np.array,
        style_img: np.array,
        loss_weights: Tuple[float],
        init_img: tfe.Variable
        ) -> tf.Tensor:

        content_rep, style_rep = self.feature_representations(content_img, style_img)
        gram_style_features = [gram_matrix(style_feature) for style_feature in style_rep]
        losses = self._loss(loss_weights, init_img, gram_style_features, content_rep)

        # TODO: of course this is not ok.
        # 1) How to prevent division by 0? Add very small number to denominator?
        # 2) What if the numerator is zero? If number is very small maybe we default to 1?
        # 3) I'd rather return a float, but can't find a neat way to convert a tf.Tensor to float
        return losses[2]/losses[1]

    def run_style_transfer(
        self,
        content_img: np.array,
        style_img: np.array,
        num_iterations=1000,
        content_weight=1.0,
        style_weight=1.0,
    ) -> Iterator[StyleTransferResult]:
        # We don't need to (or want to) train any layers of our model, so we set their
        # trainable to false.
        for layer in self._model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        content_features, style_features = self.feature_representations(
            content_img, style_img
        )
        gram_style_features = [
            gram_matrix(style_feature) for style_feature in style_features
        ]

        # Set initial image.
        init_image = self._init_image(
            content_img, from_random=self.init_image_type == InitType.RANDOM
        )

        # Create a nice config
        loss_weights = (style_weight, content_weight)

        # Compute the content2style ratio to balance losses
        c2s = self._estimate_content2weight(content_img, style_img, loss_weights, init_image)

        # update weights
        loss_weights = (1, c2s)

        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        # Store our best result
        best_loss, best_img = float("inf"), None

        # For displaying
        num_rows = 2
        num_cols = 5
        display_interval = num_iterations / (num_rows * num_cols)
        start_time = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        for i in range(num_iterations):
            grads, all_loss = self._loss_gradient(
                loss_weights, init_image, gram_style_features, content_features
            )
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                # TODO: should we return best loss and best image or should we let
                # the code do that?
                best_img = deprocess_vgg(init_image.numpy()[0])

            if i % display_interval == 0:
                start_time = time.time()

            # Use the .numpy() method to get the concrete numpy array
            plot_img = init_image.numpy()[0]
            plot_img = deprocess_vgg(plot_img)

            yield StyleTransferResult(
                image=plot_img,
                iteration_no=i,
                total_loss=loss,
                style_loss=style_score,
                content_loss=content_score,
                elapsed_time_sec=time.time() - start_time,
            )

    def _init_image(
        self, content_img: np.array, from_random: bool = False
    ) -> tfe.Variable:
        """
        initializes the image for the optimization process.
        """
        if not from_random:
            # Since in the loss function we will compare the activations
            # we need to preprocess the init_img the same way as the other image as it will
            # induce some normalizations.
            init_img = self._process_img(content_img)
        else:
            init_img = np.random.uniform(0, 255, size=content_img.shape).astype("uint8")
            init_img = scipy.ndimage.filters.median_filter(init_img, [8, 8, 1])
            init_img = self._process_img(init_img)
        return tfe.Variable(init_img, dtype=tf.float32)

    def _process_img(self, img):
        # Takes a numpy image and makes it into an image processed to be ready for vgg
        return tf.convert_to_tensor(process_vgg(img.astype("float32")), tf.float32)

    def _loss(
        self,
        loss_weights: List[float],
        init_image: tfe.Variable,
        gram_style_features: List[tf.Tensor],
        content_features: List[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """This function will compute the total loss.

        Arguments:
            loss_weights: The weights of each contribution of each loss function. 
            (style weight, content weight)
            init_image: Our initial base image. This image is what we are updating with 
            our optimization process. We apply the gradients wrt the loss we are 
            calculating to this image.
            gram_style_features: Precomputed gram matrices corresponding to the 
            defined style layers of interest.
            content_features: Precomputed outputs from defined content layers of 
            interest.

        Returns:
            returns the total loss, style loss, content loss
        """
        style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = self._model(init_image)

        num_style_layers = len(self._style_layers)
        num_content_layers = len(self._content_layers)

        style_output_features = model_outputs[:num_style_layers]
        content_output_features = model_outputs[num_style_layers:]

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        weight_per_style_layer = 1.0 / float(num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self._style_loss(
                comb_style[0], target_style
            )

        # Accumulate content losses from all layers
        weight_per_content_layer = 1.0 / float(num_content_layers)
        for target_content, comb_content in zip(
            content_features, content_output_features
        ):
            content_score += weight_per_content_layer * self._content_loss(
                comb_content[0], target_content
            )

        style_score *= style_weight
        content_score *= content_weight

        # Get total loss
        loss = style_score + content_score
        return loss, style_score, content_score

    def _content_loss(self, base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    def _style_loss(self, base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.get_shape().as_list()
        gram_style = gram_matrix(base_style)
        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )  # / (4. * (channels ** 2) * (width * height) ** 2)

    def _loss_gradient(
        self,
        loss_weights: Tuple[float, float],
        init_image: tf.Variable,
        style_features: tf.Tensor,
        content_features: tf.Tensor,
    ):
        with tf.GradientTape() as tape:
            all_loss = self._loss(
                loss_weights, init_image, style_features, content_features
            )

        total_loss = all_loss[0]
        return tape.gradient(total_loss, init_image), all_loss


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)

    # TODO: why do you divide by n? It could be the weighting factor
    return gram / tf.cast(n, tf.float32)
