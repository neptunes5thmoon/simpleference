from __future__ import print_function
import numpy as np
import scipy.ndimage


def threshold_cc(input_, offsets_wc, thr=0., 
                 ds_shape=(1, 1, 1), output_shape_wc=(1, 1, 1)):
    if not isinstance(input_, np.ndarray):
        input_ = np.array(input_)
    output = (input_ > thr).astype(np.uint64)
    scipy.ndimage.label(output, output=output)
    block_offset = [int(offsets_wc[k] // output_shape_wc[k])
                    for k in range(len(output_shape_wc))]
    block_shape = [int(np.ceil(s/float(o))) for s, o in zip(ds_shape, output_shape_wc)]
    print("Processing block at:", block_offset)
    id_offset = np.ravel_multi_index(block_offset, block_shape) * np.prod(output_shape_wc)
    output[output > 0] += id_offset
    return [input_.squeeze(), output.squeeze()]
