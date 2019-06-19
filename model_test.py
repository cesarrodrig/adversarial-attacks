
import unittest

import numpy as np

import tensorflow as tf

from model import *
from utils import get_test_image, get_test_net_config
from config import net_configs


class ModelTest(unittest.TestCase):

    def testBuildModel(self):
        net_config = get_test_net_config()
        model = Model(**net_config)

        # sanity check that all attributes are correctly set, most depend on external functions
        self.assertIsNotNone(model.graph)
        self.assertIsNotNone(model.input_tensor)
        self.assertIsNotNone(model.output_tensor)
        self.assertIsNotNone(model.output_logits_tensor)
        self.assertIsNotNone(model.session)
        self.assertIsNotNone(model.labels)

        # test the frozen graph loading
        net_config = get_test_net_config()
        net_config['frozen_graph_tar'] = "idontexist.tar"
        self.assertRaises(FileNotFoundError, Model, **net_config)

        net_config = get_test_net_config()
        net_config['frozen_graph_name'] = "idontexist.pb"
        self.assertRaises(GraphNotFoundError, Model, **net_config)


    def testIncorrectTensorNames(self):
        net_config = get_test_net_config()
        net_config['input_tensor_name'] = 'another_input:0'
        self.assertRaises(KeyError, Model, **net_config)

        net_config = get_test_net_config()
        net_config['output_tensor_name'] = 'another_output:0'
        self.assertRaises(KeyError, Model, **net_config)

        net_config = get_test_net_config()
        net_config['output_logits_tensor_name'] = 'another_output:0'
        self.assertRaises(KeyError, Model, **net_config)


    def testBuildLabels(self):
        net_config = get_test_net_config()
        model = Model(**net_config)
        self.assertEqual(len(model.labels), 1001)
        self.assertEqual(model.labels[0], 'background')

        net_config['labels_file'] = "idontexist.txt"
        self.assertRaises(FileNotFoundError, Model, **net_config)


    def testCallableEndToEnd(self):
        net_config = get_test_net_config()
        model = Model(**net_config)

        image = get_test_image()
        probs = model([image])
        self.assertEqual(len(probs), 1)
        labels_i = np.argmax(probs, axis=1)
        self.assertEqual(len(labels_i), 1)
        label = model.label_to_label_name(labels_i[0])
        self.assertEqual(label, 'digital clock')


if __name__ == '__main__':
    unittest.main()
