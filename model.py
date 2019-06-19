import os
import tarfile

import tensorflow as tf

import numpy as np

"""
You don't want to have an attack method inside your model.
Your model should only be capable of being called and provide information about the model.
Your attack should be capable of taking the attributes of the model and attacking it.
"""

class GraphNotFoundError(Exception):
    pass

class Model(object):


    def __init__(self, frozen_graph_tar="",
        frozen_graph_name="frozen_inference_graph.pb",
        input_tensor_name='input:0',
        output_tensor_name='output:0',
        output_logits_tensor_name='output_logits:0',
        labels_file="models/labels.txt"):

        """Creates and loads a frozen graph model.
        Args:
            frozen_graph_tar:           filename of the tar file containing the frozen graph.
            frozen_graph_name:          name of the frozen graph to load from the tar file.
            input_tensor_name:          name of the tensor that receives the input image.
            output_tensor_name:         name of the tensor that outputs the softmax probabilities.
            output_logits_tensor_name:  name of the tensor that outputs the logits that go through the softmax layer.
            labels_file:                filename of the file containing the list of labels. Labels are assumed to be separated by a newline.

        Returns:
            a callable Model instance with the given frozen graph.
        """

        graph_def = None
        # Extract frozen graph from tar archive.
        with tarfile.open(frozen_graph_tar) as tar_file:
            for tar_info in tar_file.getmembers():
                if frozen_graph_name in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.GraphDef.FromString(file_handle.read())
                    break

        if graph_def is None:
            raise GraphNotFoundError('Could not load graph')

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.session = tf.Session(graph=self.graph)

        self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)
        self.output_logits_tensor = self.graph.get_tensor_by_name(output_logits_tensor_name)

        with open(labels_file, 'r') as f:
            self.build_labels(f)


    def __call__(self, x):
        """Runs inference on one or more images.

        Args:
          x:             input image(s)

        Returns:
          A list with the label probabilities
        """

        feed_dict = {self.input_tensor: x }
        return self.session.run(self.output_tensor, feed_dict=feed_dict)


    def label_to_label_name(self, label):
        """
        Converts a label index to a label name

        Args:
            label:  index number of the label

        Returns:
            label_str:  the label name
        """
        return self.labels[label]


    def build_labels(self, labels):
        """
        Builds the labels map from index to name

        Args:
            labels: list or iterable of labels

        Returns:
            None
        """
        self.labels = {i: line.strip() for i, line in enumerate(labels) }
