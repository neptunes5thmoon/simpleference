
def normalize(data, eps=1e-4, mean=None, std=None, filter_zeros=True):
    if filter_zeros:
        data_pre = data[data != 0]
    else:
        data_pre = data
    mean = data_pre.mean() if mean is None else mean
    std = data_pre.std() if std is None else std
    return (data - mean) / (std + eps)


def cast(data, dtype='float32'):
    return data.astype(dtype, copy=False)


def preprocess(data, mean=None, std=None):
    return normalize(cast(data), mean=mean, std=std)
