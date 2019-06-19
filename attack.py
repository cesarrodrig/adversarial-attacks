
import tensorflow as tf

class UnsupportedModel(Exception):
    pass


class Attack(object):
    """
      Abstract base class for the attack classes.
    """

    REQUIRED_MODEL_ATTRS = ['input_tensor', 'output_tensor', 'output_logits_tensor']

    def __init__(self, model):
        """ Initializes the Attack instance.
        Args:
            model: instance of model.Model or similar with attributes 'input_tensor', 'output_tensor' and 'output_logits_tensor'

        Raises:
            'UnsupportedModel' if model does not contain all the required attributes.
        """

        for required_attr in self.REQUIRED_MODEL_ATTRS:
            if not hasattr(model, required_attr):
                raise UnsupportedModel("model does not contain the attribute '%s'" % required_attr)

        self.model = model

    def __call__(self, x, epsilon=0.01, clip_min=0., clip_max=1.):
        raise NotImplementedError(str(type(self)) + " must implement `__call__`.")


class FGSM(Attack):

    def __call__(self, x, epsilon=0.01, clip_min=0., clip_max=1.):
        """
        Runs one iteration of FGSM on x.

        Args:
            x:          input tensor from which the loss is calculated from.
            epsilon:    the per-pixel alteration done on the adversarial image.
            clip_min:   the minimum value of the output image pixel.
            clip_max:   the maximum value of the output image pixel.
        Return:
            x_adv: the adversarial input altered by one step.
        """
        epsilon = tf.abs(epsilon)

        # get the softmax probabilities and logit values which produced them
        probs = self.model.output_tensor
        logits = self.model.output_logits_tensor
        yshape = probs.get_shape().as_list()
        ydim = yshape[1]

        # get the predicted classes
        indices = tf.argmax(probs, axis=1)
        labels = tf.one_hot(indices, ydim)

        # calculate the loss of the current image predicted class
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

        # calculate the gradient of the loss with respect to the input
        dy_dx, = tf.gradients(loss, x)
        # alter the image pixels by epsilon
        x_adv = tf.stop_gradient(x + epsilon*tf.sign(dy_dx))
        # ensure the pixels are within range
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        return x_adv
