SHELL := /bin/bash

all: clean setup onnx keras test_onnx test_keras
clean:
	rm -rf *.h5 *.json *.params *.onnx *_tfjs __pycache__ *.pytorch
setup: clean
	virtualenv venv
	source venv/bin/activate && pip install -r requirements.txt
onnx: 
	python3 export_gluon_to_onnx.py
keras: 
	python3 export_onnx_to_keras.py
test_onnx:
	python3 inference_onnx.py
test_keras:
	python3 inference_keras.py


