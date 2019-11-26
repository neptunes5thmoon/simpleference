from __future__ import print_function
import h5py
import z5py
import os
import json
from random import shuffle
import numpy as np
import re
import fnmatch
from .io import IoN5
from .inference import load_input_crop
import dask
import toolz as tz
import logging


def _offset_list(shape, output_shape):
    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                in_list.append([z, y, x])
    return in_list


# NOTE this will not cover the whole volume
def _offset_list_with_shift(shape, output_shape, shift):
    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                in_list.append([min(z + shift[0], shape[0]),
                                min(y + shift[1], shape[1]),
                                min(x + shift[2], shape[2])])
    return in_list


# this returns the offsets for the given output blocks.
# blocks are padded on the fly during inference if necessary
def get_offset_lists(shape,
                     gpu_list,
                     save_folder,
                     output_shape,
                     randomize=False,
                     shift=None):
    in_list = _offset_list(shape, output_shape) if shift is None else\
            _offset_list_with_shift(shape, output_shape, shift)
    if randomize:
        shuffle(in_list)

    n_splits = len(gpu_list)
    out_list = [in_list[i::n_splits] for i in range(n_splits)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_%i.json' % gpu_list[ii])
        with open(list_name, 'w') as f:
            json.dump(olist, f)


# this returns the offsets for the given output blocks and bounding box.
# blocks are padded on the fly during inference if necessary
def get_offset_lists_with_bb(shape,
                             gpu_list,
                             save_folder,
                             output_shape,
                             bb_start,
                             bb_stop,
                             randomize=False):

    # zap the bounding box to grid defined by out_blocks
    bb_start_c = [(bbs // outs) * outs for bbs, outs in zip(bb_start, output_shape)]
    bb_stop_c = [(bbs // outs + 1) * outs for bbs, outs in zip(bb_stop, output_shape)]

    in_list = []
    for z in range(bb_start_c[0], bb_stop_c[0], output_shape[0]):
        for y in range(bb_start_c[1], bb_stop_c[1], output_shape[1]):
            for x in range(bb_start_c[2], bb_stop_c[2], output_shape[2]):
                in_list.append([z, y, x])

    if randomize:
        shuffle(in_list)

    n_splits = len(gpu_list)
    out_list = [in_list[i::n_splits] for i in range(n_splits)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_%i.json' % gpu_list[ii])
        with open(list_name, 'w') as f:
            json.dump(olist, f)


# redistributing offset lists from failed jobs
def redistribute_offset_lists(gpu_list, save_folder):
    p_full = re.compile("list_gpu_\d+.json")
    p_proc = re.compile("list_gpu_\d+_\S*_processed.txt")
    full_list_jsons = []
    processed_list_files = []
    for f in os.listdir(save_folder):
        mo_full = p_full.match(f)
        mo_proc = p_proc.match(f)
        if mo_full is not None:
           full_list_jsons.append(f)
        if mo_proc is not None:
           processed_list_files.append(f)
    full_block_list = set()
    for fl in full_list_jsons:
        with open(os.path.join(save_folder, fl), 'r') as f:
            bl = json.load(f)
            full_block_list.update({tuple(coo) for coo in bl})
    processed_block_list = set()
    bls = []
    for pl in processed_list_files:
        with open(os.path.join(save_folder, pl), 'r') as f:
            bl_txt = f.read()
        bl_txt = '[' + bl_txt[:bl_txt.rfind(']') + 1] + ']'
        bls.append(json.loads(bl_txt))
        processed_block_list.update({tuple(coo) for coo in bls[-1]})

    to_be_processed_block_list = list(full_block_list - processed_block_list)
    previous_tries = []
    p_tries = re.compile("list_gpu_\d+_try\d+.json")
    for f in os.listdir(save_folder):
        mo_tries = p_tries.match(f)
        if mo_tries is not None:
            previous_tries.append(f)

    if len(previous_tries) == 0:
        tryno = 0
    else:
        trynos = []
        for tr in previous_tries:
            trynos.append(int(tr.split('try')[1].split('.json')[0]))
        tryno = max(trynos)+1
    print('Backing up last try ({0:})'.format(tryno))
    for f in full_list_jsons:
        os.rename(os.path.join(save_folder,f), os.path.join(save_folder, f[:-5] + '_try{0:}.json'.format(tryno)))
    for f in processed_list_files:
        os.rename(os.path.join(save_folder,f), os.path.join(save_folder, f[:-4] + '_try{0:}.txt'.format(tryno)))
    n_splits = len(gpu_list)
    out_list = [to_be_processed_block_list[i::n_splits] for i in range(n_splits)]
    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_%i.json' % gpu_list[ii])
        with open(list_name, 'w') as f:
            json.dump(olist, f)

def load_mask(path, key):
    ext = os.path.splitext(path)[-1]
    if ext.lower() in ('.h5', '.hdf', '.hdf'):
        with h5py.File(path, 'r') as f:
            mask = f[key]
    elif ext.lower() in ('.zr', '.zarr', '.n5'):
        with z5py.File(path, 'r') as f:
            mask = f[key]
    return mask


def generate_list_for_mask(offset_file_json, output_shape_wc, path, mask_ds, n_cpus):
    mask = load_mask(path, mask_ds)
    mask_voxel_size = mask.attrs["pixelResolution"]["dimensions"]
    shape_wc = tuple(np.array(mask.shape) * np.array(mask_voxel_size))
    complete_offset_list = _offset_list(shape_wc, output_shape_wc)

    io = IoN5(path, mask_ds, voxel_size=mask_voxel_size, channel_order =None)

    @dask.delayed()
    def load_offset(offset_wc):
        return load_input_crop(io, offset_wc, (0,) * len(output_shape_wc), output_shape_wc, padding_mode="constant")[0]

    @dask.delayed()
    def evaluate_mask(mask_block):
        if np.sum(mask_block) > 0:
            return True
        else:
            return False

    offsets_mask_eval = []
    for offset_wc in complete_offset_list:
        keep_offset = tz.pipe(offset_wc, load_offset, evaluate_mask)
        offsets_mask_eval.append((offset_wc, keep_offset))

    offsets_mask_eval = dask.compute(*offsets_mask_eval, scheduler="threads", num_workers=n_cpus)

    offsets_in_mask = []
    for o, m in offsets_mask_eval:
        if m:
            offsets_in_mask.append(o)

    logging.info("{0:}/{1:} blocks contained in mask, saving offsets in {2:}".format(len(offsets_in_mask),
                                                                                     len(complete_offset_list),
                                                                                     offset_file_json))
    with open(offset_file_json, 'w') as f:
        json.dump(offsets_in_mask, f)


# this returns the offsets for the given output blocks.
# blocks are padded on the fly in the inference if necessary
def offset_list_from_precomputed(input_list,
                                 gpu_list,
                                 save_folder,
                                 list_name_extension='',
                                 randomize=False):

    if isinstance(input_list, str):
        with open(input_list, 'r') as f:
            input_list = json.load(f)
    else:
        assert isinstance(input_list, list)

    if randomize:
        shuffle(input_list)

    n_splits = len(gpu_list)
    out_list = [input_list[i::n_splits] for i in range(n_splits)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    print("Original len", len(input_list))
    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_{0:}{1:}.json'.format(gpu_list[ii], list_name_extension))
        print("Dumping list number", ii, "of len", len(olist))
        with open(list_name, 'w') as f:
            json.dump(olist, f)


def stitch_prediction_blocks(save_path,
                             block_folder,
                             shape,
                             key='data',
                             end_channel=None,
                             n_workers=8,
                             chunks=(1, 64, 64, 64)):
    from concurrent import futures
    if end_channel is None:
        chan_slice = (slice(None),)
    else:
        assert end_channel <= shape[0]
        chan_slice = (slice(0, end_channel),)

    def stitch_block(ds, block_id, block_file, n_blocks):
        print("Stitching block %i / %i" % (block_id, n_blocks))
        offsets = [int(off) for off in block_file[:-3].split('_')[1:]]
        with h5py.File(os.path.join(block_folder, block_file), 'r') as g:
            block_data = g['data'][:]
        block_shape = block_data.shape[1:]
        # Need to add slice for channel dimension
        bb = chan_slice + tuple(slice(off, off + block_shape[ii])
                                for ii, off in enumerate(offsets))
        ds[bb] = block_data

    with h5py.File(save_path, 'w') as f:
        ds = f.create_dataset(key,
                              shape=shape,
                              dtype='float32',
                              compression='gzip',
                              chunks=chunks)
        files = os.listdir(block_folder)
        # filter out invalid filenames
        files = [ff for ff in files if ff.startswith('block')]
        # make sure all blocks are h5 files
        assert all(ff[-3:] == '.h5' for ff in files)
        n_blocks = len(files)
        with futures.ThreadPoolExecutor(max_workers=n_workers) as tp:
            tasks = [tp.submit(stitch_block, ds, block_id, block_file, n_blocks)
                     for block_id, block_file in enumerate(files)]
            [t.result() for t in tasks]


def extract_nn_affinities(save_prefix,
                          block_folder,
                          shape,
                          invert_affs=False):
    from concurrent import futures
    save_path_xy = save_prefix + '_xy.h5'
    save_path_z = save_prefix + '_z.h5'
    with h5py.File(save_path_xy, 'w') as f_xy, h5py.File(save_path_z, 'w') as f_z:
        ds_xy = f_xy.create_dataset('data',
                                    shape=shape,
                                    dtype='float32',
                                    compression='gzip',
                                    chunks=(56, 56, 56))
        ds_z = f_z.create_dataset('data',
                                  shape=shape,
                                  dtype='float32',
                                  compression='gzip',
                                  chunks=(56, 56, 56))
        files = os.listdir(block_folder)

        def extract_block(i, ff):
            print("Stitching block %i / %i" % (i, len(files)))
            offsets = [int(off) for off in ff[:-3].split('_')[1:]]

            with h5py.File(os.path.join(block_folder, ff), 'r') as g:
                block_data = g['data'][:3]

            if invert_affs:
                block_data = 1. - block_data

            block_shape = block_data.shape[1:]
            # Need to add slice for channel dimension
            bb = tuple(slice(off, off + block_shape[ii]) for ii, off in enumerate(offsets))
            ds_xy[bb] = (block_data[1] + block_data[2]) / 2.
            ds_z[bb] = block_data[0]

        with futures.ThreadPoolExecutor(max_workers=20) as tp:
            tasks = []
            for i, ff in enumerate(files):
                if not ff.startswith('block'):
                    continue
                assert ff[-3:] == '.h5'
                tasks.append(tp.submit(extract_block, i, ff))
            [t.result() for t in tasks]


def reject_empty_batch(data):
    return np.sum(data) == 0
