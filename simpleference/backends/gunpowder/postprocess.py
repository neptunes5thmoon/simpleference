from __future__ import print_function
import numpy as np
import scipy.ndimage


def threshold_cc(data, output_bounding_box, thr=0., ds_shape=(1, 1, 1), output_shape=(1, 1, 1)):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    output = (data > thr).astype(np.uint64)
    scipy.ndimage.label(output, output=output)
    block_offset = [int(output_bounding_box[k].start // output_shape[k]) for k in range(len(output_shape))]
    block_shape = [int(np.ceil(s/float(o))) for s, o in zip(ds_shape, output_shape)]
    print("Processing block at:", block_offset)
    id_offset = np.ravel_multi_index(block_offset, block_shape)*np.prod(output_shape)
    output[output > 0] += id_offset
    return [data.squeeze(), output.squeeze()]


def nn_affs(data, output_bounding_box):
    output = np.empty(shape=(2,)+data.shape[1:], dtype=data.dtype)
    output[0] = (data[1] + data[2]) / 2.
    output[1] = data[0]
    return output


def clip_float_to_uint8(data, output_bounding_box, float_range=(0, 1), safe_scale=True):
    """Convert float values in the range float_range to uint8. Crop values to (0, 255).

    Args:
        data (np.array): Input data as produced by the network.
        output_bounding_box (slice): Bounding box of the current block in the full dataset.
        float_range (tuple, list): Range of values of data.
        safe_scale (bool): If True, values are scaled such that all values within float_range fall within (0, 255).
            and are not cropped.  If False, values at the lower end of float_range may be scaled to < 0 and then
            cropped to 0.

    Returns:
        np.array: Postprocessed output, dtype is uint8
    """
    print(list(output_bounding_box[k].start for k in range(len(output_bounding_box))))
    if safe_scale:
        mult = np.floor(255./(float_range[1]-float_range[0]))
    else:
        mult = np.ceil(255./(float_range[1]-float_range[0]))
    add = 255 - mult*float_range[1]
    return np.clip((data*mult+add).round(), 0, 255).astype('uint8')
