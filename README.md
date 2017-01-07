# tensorflow-helpers
This repository contains different helpers for building TensorFlow models

## Installation
1. Activate your virtual environment: `source ~/<path_to_venv>/bin/activate`
2. (Optional) Update your TensorFlow installation: 
`pip uninstall tensorflow`, `pip install -U tensorflow-gpu`
3. Install the package: `pip install -U git+git://github.com/text-machine-lab/tensorflow-helpers.git`

## Usage
This package provides a base class `BaseModel` which serves as a starting point 
for building you model.
The main work to be done while using this class is to override the `build_model` method, 
in which you should set the attributes `self.op_loss` and `self.op_predict`. 
The loss function, provided by the former attribute will be optimized 
when you call the `model` method. The latter attribute will be used 
to provide the results when you call the `predict` method.

After that, you can add inputs to the model by calling the `add_input` method and, finally, 
train the model using the `train_model` method.

### Example
See an example here: 
[examples/simple_model.ipynb](https://github.com/text-machine-lab/tensorflow-helpers/examples/simple_model.ipynb)

