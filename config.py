
"""
Net tars downloaded from:
    https://www.tensorflow.org/lite/guide/hosted_models

"""
net_configs = {

    # Top-1: 82.6% Top-5: 96.1%
    'nasnet_large': {
        'frozen_graph_tar':  'models/nasnet_large_2018_04_27.tgz',
        'frozen_graph_name':  'nasnet_large.pb',
        'input_tensor_name':  'input:0',
        'output_tensor_name':  'final_layer/predictions:0',
        'output_logits_tensor_name':  'final_layer/FC/BiasAdd:0',
        'labels_file': 'models/nasnet_labels.txt',
        'image_size': 331,
    },

    # Top-1 70.1% Top-5: 89.8%
    'inception_v1_quant' : {
        'frozen_graph_tar':  'models/inception_v1_224_quant_20181026.tgz',
        'frozen_graph_name':  'inception_v1_224_quant_frozen.pb',
        'input_tensor_name':  'input:0',
        'output_tensor_name':  'InceptionV1/Logits/Predictions/Softmax:0',
        'output_logits_tensor_name':  'InceptionV1/Logits/Predictions/Reshape:0',
        'image_size': 224,
    },

}
