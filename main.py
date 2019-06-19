
import random

import click

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec

import tensorflow as tf

from attack import FGSM
from model import Model
from utils import resize_image, load_image
from config import net_configs

seed = 1
tf.set_random_seed(seed)
random.seed(seed)

# tested with 'inception_v1_quant'
IMAGES = {
    # 7 steps required to succeed
    'digital_clock_1': 'https://www.maxiaids.com/media/thumbs/0001/0001249-jumbo-display-digital-alarm-clock-with-3-inch-led.jpg',
    # 7 steps required to succeed
    'digital_clock_2': 'https://ae01.alicdn.com/kf/HTB15y_cIFXXXXaEXpXXq6xXFXXXd/Baldr-Modern-Student-Alarm-font-b-Clock-b-font-Repeating-Snooze-Light-font-b-Digital-b.jpg',
    # 7 steps required to succeed
    'digital_clock_3': 'https://target.scene7.com/is/image/Target/14193190',
    # 1 step for similar but incorrect breed, 2 steps for low confidence wrong guess.
    'dog_1' : 'https://www.wikihow.com/images/6/64/Stop-a-Dog-from-Jumping-Step-6-Version-2.jpg',
    # 2 steps required to succeed
    'beer_1' : 'https://img-new.cgtrader.com/items/20608/heineken_beer_bottle_3d_model_fbx_lwo_lw_lws_obj_max_lxo_79aea18d-613b-46ab-b323-2bc0c3a740dc.jpg',
    # 2 steps required to succeed
    'cassette_1' : 'https://img1.southernliving.timeinc.net/sites/default/files/styles/4_3_horizontal_-_1200x900/public/1542230813/GettyImages-172757757.jpg?itok=2l04nPOg',
}

IMAGE_NAME = 'cassette_1'
NET_NAME = 'inception_v1_quant' # 'nasnet_large'
STEPS = 1


def perform_attack(model, attack, image, steps):
    """ Performs the given attack in model using image.
    Args:
        model:  Model instance or similar with the attributes 'input_tensor', 'output_tensor' and 'output_logits_tensor'
        attack: Attack instance or similar callable that receives an input tensor and returns an adversarial output tensor.
        image:  np.ndarray representing an image.
        steps:  int for the number of steps to perform the attack.

    Returns:
        An np.ndarray with the resulting adversarial image.
    """

    with model.graph.as_default():
        with tf.Session() as sess:

            adv = [image]
            x_adv = attack(model.input_tensor, clip_max=1.0)

            for i in range(steps):
                feed_dict = {model.input_tensor: adv}
                adv = sess.run(x_adv, feed_dict=feed_dict)

            return adv[0]


def predict(model, images):
    """ Predicts the labels of the images using model
    Args:
        model:  Model instance or similar.
        images: list or iterable of np.ndarray containing the images.

    Returns:
        The predicted label names and their probabilities.
    """
    probs = model(images)
    labels = np.argmax(probs, axis=1)
    labels = [ model.label_to_label_name(label) for label in labels ]
    return labels, probs


def plot_images(original_img, adv_img, labels, probs):
    """ Uses matplotlib to plot the original image, the approximated adversarial mask and the adversarial image, along with their labels and probabilities.
    Args:
        original_img:   np.ndarray with the original image
        adv_img:        np.ndarray with the adversarial image
        labels:         list of 2 with the labels of the original and adversarial image respectively.
        probs:          list of 2 with the probabilities of the original and adversarial label respectively.
    """
    original_label, adv_label = labels
    original_prob, adv_prob = probs

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2])

    plt.subplot(grid_spec[0])
    plt.imshow(original_img)
    plt.axis('off')
    plt.title('Input image was classified as: %s with %0.3f' % (original_label, original_prob) )

    plt.subplot(grid_spec[1])
    plt.imshow(original_img - adv_img + 0.5)
    plt.axis('off')
    plt.title('adversarial mask')

    plt.subplot(grid_spec[2])
    plt.imshow(adv_img)
    plt.axis('off')
    plt.title('Adversarial image was classified as: %s with %0.3f' % (adv_label, adv_prob ) )

    plt.show()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-u', '--image-url', 'image_url', type=str, default="",
    help="URL containing an image to be used in the attack. Leave empty to choose by image name.", show_default=False)

@click.option('-i', '--image-name', 'image_name', type=str, default=IMAGE_NAME,
    help="Image to be used in the attack.", show_default=True)

@click.option('-n', '--net-name', 'net_name', type=click.Choice(net_configs.keys()), default=NET_NAME,
    help="Neural Network to use.", show_default=True)

@click.option('-s', '--steps', 'steps', type=int, default=STEPS,
    help="Number of steps of Fast Gradient Sign Method to perform.", show_default=True)
def run_command(net_name, image_name, image_url, steps):
    """
    Scripts that takes an image classification neural network and attacks it via FGSM with a selected image.
    After performing the attack, the input image, the adversarial mask and the adversarial image are displayed.
    """
    net_config = net_configs[net_name]
    image_size = net_config.pop('image_size')
    if image_url:
        image = load_image(image_url, image_size=image_size)
    else:
        image = load_image(IMAGES[image_name], image_size=image_size)

    try:
        model = Model(**net_config)
        fgsm = FGSM(model)
        adv_img = perform_attack(model, fgsm, image, steps)
        labels, probs = predict([image, adv_img])
        plot_images(image, adv_img, labels, probs)
    except Exception as e:
        print("An error occurred: %s" % repr(e))
        exit(1)


if __name__ == '__main__':
    run_command()
