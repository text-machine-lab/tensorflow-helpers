import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow_helpers.models.base_model import BaseModel


class MultiplyModel(BaseModel):
    def __init__(self):
        super(MultiplyModel, self).__init__()

    def build_model(self):
        a = self.input_dict['a']
        b = self.input_dict['b']
        self.op_predict = a * b


def test_input_data_dict():
    model = MultiplyModel()

    model.add_input('a', [], dtype=tf.int32)
    model.add_input('b', [], dtype=tf.int32)

    model.build_model()

    a = np.array([2, 1])
    b = np.array([4, 3])

    data_dict = {
        'a': a,
        'b': b,
    }

    result = model.predict(data=data_dict)

    assert result[0] == 8
    assert result[1] == 3

def test_input_kwargs():
    model = MultiplyModel()

    model.add_input('a', [], dtype=tf.int32)
    model.add_input('b', [], dtype=tf.int32)

    model.build_model()


    a = np.array([2, 1])
    b = np.array([4, 3])

    result = model.predict(a=a, b=b)

    assert result[0] == 8
    assert result[1] == 3
