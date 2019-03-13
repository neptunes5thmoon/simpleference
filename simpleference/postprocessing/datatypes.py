import numpy as np


def clip_float_to_uint8(input_, output_bounding_box, float_range=(0., 1.), safe_scale=True):
    """Convert float values in the range float_range to uint8. Crop values to (0, 255).

    Args:
        input_ (np.array): Input data as produced by the network.
        output_bounding_box (slice): Bounding box of the current block in the full dataset.
        float_range (tuple, list): Range of values of data.
        safe_scale (bool): If True, values are scaled such that all values within float_range fall within (0, 255).
            and are not cropped.  If False, values at the lower end of float_range may be scaled to < 0 and then
            cropped to 0.

    Returns:
        np.array: Postprocessed output, dtype is uint8
    """
    def clip_float_to_uint8_arr(input_, output_bb, float_range=(0.,1.), safe_scale=True):
        assert isinstance(input_, np.ndarray)
        if safe_scale:
            mult = np.floor(255./(float_range[1]-float_range[0]))
        else:
            mult = np.ceil(255./(float_range[1]-float_range[0]))
        add = 255 - mult*float_range[1]
        return np.clip((input_*mult+add).round(), 0, 255).astype('uint8')
    if isinstance(input_, list):
        ret = []
        if not isinstance(output_bounding_box, list):
            output_bounding_box = [output_bounding_box]*len(input_)

        for i, obb in zip(input_, output_bounding_box):
            ret.append(clip_float_to_uint8_arr(i, obb, float_range=float_range, safe_scale=safe_scale))
        return ret

    elif isinstance(input_, np.ndarray):
        return clip_float_to_uint8_arr(input_, output_bounding_box, float_range=float_range, safe_scale=safe_scale)
    else:
        raise TypeError
