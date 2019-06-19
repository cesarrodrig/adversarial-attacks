# FGSM Adversarial Attack On Image Classification

## Requirements

	pip install -r requirements.txt

The configurations for nasnet and inception_v1 are already in `config.py` and others can be added.
The net tars have to be downloaded from [here](https://www.tensorflow.org/lite/guide/hosted_models) and
placed in the `models` folder.

## Usage

To run:

	python main.py

For help:

	python main.py -h

To run the tests:

	python -m unittest discover -s . -p '*_test.py'
