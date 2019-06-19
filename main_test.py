
import unittest

import numpy as np

from main import *
from model import Model
from attack import FGSM
from utils import get_test_net_config, get_test_image


class PerformAttackTest(unittest.TestCase):

    def testEndToEnd(self):
        model = Model(**get_test_net_config())
        attack = FGSM(model)
        image = get_test_image()
        steps = 1
        adv = perform_attack(model, attack, image, steps)
        self.assertEqual(image.shape, adv.shape)
        self.assertFalse(np.array_equal(image, adv))


class PredictTest(unittest.TestCase):

    def testEndToEnd(self):
        model = Model(**get_test_net_config())
        image = get_test_image()

        labels, probs = predict(model, [image])
        self.assertTrue(len(labels), 1)
        self.assertTrue(len(probs), 1)
        self.assertEqual(labels[0], "digital clock")


if __name__ == '__main__':
    unittest.main()
