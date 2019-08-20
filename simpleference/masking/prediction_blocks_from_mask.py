from __future__ import print_function, division
import os
from math import ceil
import json
import numpy as np

import h5py
import z5py


def load_mask(path, key):
    ext = os.path.splitext(path)[-1]
    if ext.lower() in ('.h5', '.hdf', '.hdf'):
        with h5py.File(path, 'r') as f:
            mask = f[key][:]
    elif ext.lower() in ('.zr', '.zarr', '.n5'):
        with z5py.File(path, 'r') as f:
            mask = f[key][:]
    return mask


def make_prediction_blocks(shape, output_block_shape, mask_file, output_file,
                           mask_key='data', roi_begin=None, roi_end=None):

    # load mask and get mask shape
    assert os.path.exists(mask_file), mask_file
    mask = load_mask(mask_file, mask_key)
    mask_shape = mask.shape

    # compute output block shape
    blocked_shape = [int(ceil(float(fs) / os)) for fs, os in zip(shape, output_block_shape)]

    # compute the scale factors between mask and blocked_shape
    scale_factor_mask = [float(sh) / bs for sh, bs in zip(mask_shape, blocked_shape)]

    if roi_begin is not None:
        assert roi_end is not None
        roi_begin_out = [roib // obs for roib, obs in zip(roi_begin, output_block_shape)]
        roi_end_out = [roie // obs + 1 for roie, obs in zip(roi_end, output_block_shape)]
    else:
        roi_begin_out = [0, 0, 0]
        roi_end_out = blocked_shape

    prediction_mask = np.zeros(blocked_shape, dtype='uint8')
    prediction_blocks = []
    # generate blocks
    for z in range(roi_begin_out[0], roi_end_out[0]):
        print("generating for", z, "/", roi_end_out[0])
        for y in range(roi_begin_out[1], roi_end_out[1]):
            for x in range(roi_begin_out[2], roi_end_out[2]):
                coord = [z, y, x]
                bb_mask = tuple(slice(int(co * sf),
                                      int(ceil((co + 1) * sf)))
                                for co, sf in zip(coord, scale_factor_mask))
                mask_block = mask[bb_mask]

                if np.sum(mask_block) > 0:
                    coord_data = [co * sf for co, sf in zip(coord, output_block_shape)]
                    prediction_blocks.append(coord_data)
                    prediction_mask[z, y, x] = 1

    with open(output_file, 'w') as f:
        json.dump(prediction_blocks, f)

    n_blocks_total = np.prod(blocked_shape)
    print("Number of blocks to predict", len(prediction_blocks))
    print("Percentage of blocks that will be predicted", len(prediction_blocks) / n_blocks_total)

    return prediction_mask


def order_blocks(block_file, out_file, central_coordinate, resolution=(1, 1, 1)):
    assert isinstance(central_coordinate, np.ndarray)
    assert len(central_coordinate) == 3
    with open(block_file, 'r') as f:
        prediction_blocks = np.array(json.load(f))

    distances = np.sum(np.square(np.multiply(prediction_blocks, resolution) -
                                 np.multiply(central_coordinate, resolution)), axis=1)
    sort = np.argsort(distances)

    prediction_blocks = prediction_blocks[sort]

    with open(out_file, 'w') as f:
        json.dump(prediction_blocks.tolist(), f)
