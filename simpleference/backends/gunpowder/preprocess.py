import numpy as np


def multiple_inputs(func):
    def wrapper(data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            return func(data, *args, **kwargs)
        else:
            return [func(d, *args, **kwargs) for d in data]
    return wrapper

@multiple_inputs
def normalize(data):
    if data.dtype == np.uint8:
        factor = 1. / 255
    elif data.dtype == np.float32:
        assert data.min() >= 0 and data.max() <= 1, \
            "Raw values are float but not in [0,1], I don't know how to normalize. Please provide a factor."
        factor = 1.
    else:
        raise RuntimeError("False input dtype")

    return data * factor

@multiple_inputs
def scale_shift(data, scale, shift):
    return data * scale + shift

@multiple_inputs
def zero_out_const_sections(data):
    for z in range(data.shape[0]):
        if data[z].min() == data[z].max():
            data[z] = 0
    return data

@multiple_inputs
def preprocess(data, scale=2, shift=-1):
    return zero_out_const_sections(scale_shift(normalize(data),
                                                         scale,
                                                         shift))
