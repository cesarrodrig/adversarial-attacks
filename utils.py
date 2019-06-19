
from six.moves import urllib
from io import BytesIO

import numpy as np
from PIL import Image

from config import net_configs

def resize_image(image, to_size):
    """
    Resizes and image to to_size. The resulting image has square dimensions.

    Args:
        image:      PIL image instance
        to_size:    int with the final number of height and width

    Returns:
        PIL resized image.
    """
    width, height = image.size
    width_resize_ratio = 1.0 * to_size / width
    height_resize_ratio = 1.0 * to_size / height
    target_size = (int(width_resize_ratio * width), int(height_resize_ratio * height))
    return image.convert('RGB').resize(target_size, Image.ANTIALIAS)

def load_image(url, image_size=None):
    """
    Loads an image from a URL and resizes it if image_size is given.

    Args:
        url:        string with the URL where the image should be downloaded from.
        image_size: int with the desired dimensions of the image.

    Returns:
        A np.array representing the image and pixels with range [0, 1].
    """
    try:
        f = urllib.request.urlopen(url)
        jpeg_str = f.read()
        image = Image.open(BytesIO(jpeg_str))
        if image_size:
            image = resize_image(image, to_size=image_size)

        return np.asarray(image)/255.

    except IOError:
        print('Cannot load image. Please check url: ' + url)
        return


""" Test-related functions"""

def get_test_net_config():
    net_config = net_configs['inception_v1_quant'].copy()
    net_config.pop('image_size')
    return net_config


def get_test_image():
    image_url = 'https://www.maxiaids.com/media/thumbs/0001/0001249-jumbo-display-digital-alarm-clock-with-3-inch-led.jpg'
    return load_image(image_url, image_size=224)
