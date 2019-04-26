"""Utilities to generate a dataset for text recognition"""
import glob
from PIL import Image, ImageDraw, ImageFont


DEFAULT_VOCABULARY = ["Hello", "world"]


def _all_fonts():
    # Return all truetype fonts on linux systems
    return sorted(glob.iglob("/usr/share/fonts/**/*.ttf", recursive=True))


class DatasetGenerator:
    def __init__(self, vocabulary=None):
        self._font_list = _all_fonts()
        self._vocabulary = DEFAULT_VOCABULARY

    def generate_image(self, width, height, seed=None):

        text_color = (0, 0, 0)
        bg_color = (255, 255, 255)
        text = self._vocabulary[0]
        font_file = self._font_list[0]
        font_size = 45
        text_rotation = 5

        # Draw font with correct size
        font = ImageFont.truetype(font_file, font_size)
        text_width, text_height = font.getsize(text)
        img = Image.new("RGB", (text_width, text_height), color=bg_color)
        d = ImageDraw.Draw(img)

        d.text((0, 0), text, font=font, fill=text_color)

        # Rotate
        img = img.rotate(text_rotation, fillcolor=bg_color)

        # Scale image to correct size
        img = img.resize((width, height))
        return img

