from __future__ import print_function
import os
import json

import numpy as np
import dask
import toolz as tz
import functools

from .io import IoN5, IoHDF5  # IoDVID


def load_input(ios, offset_wc, contexts_wc, output_shape_wc, padding_mode='reflect'):
    if isinstance(ios, tuple) or isinstance(ios, list):
        assert (isinstance(contexts_wc[0], tuple) or isinstance(contexts_wc[0], list))
        assert len(contexts_wc) == len(ios)
    else:
        ios = (ios, )
        contexts_wc = (contexts_wc, )
    datas  = []

    for io, context_wc in zip(ios, contexts_wc):

        starts_wc = [off_wc - context_wc[i] for i, off_wc in enumerate(offset_wc)]
        stops_wc = [off_wc + output_shape_wc[i] + context_wc[i] for i, off_wc in enumerate(offset_wc)]
        shape_wc = io.shape

        # we pad the input volume if necessary
        pad_left_wc = None
        pad_right_wc = None

        # check for padding to the left
        if any(start_wc < 0 for start_wc in starts_wc):
            pad_left_wc = tuple(abs(start_wc) if start_wc < 0 else 0 for start_wc in starts_wc)
            starts_wc = [max(0, start_wc) for start_wc in starts_wc]

        # check for padding to the right
        if any(stop_wc > shape_wc[i] for i, stop_wc in enumerate(stops_wc)):
            pad_right_wc = tuple(stop_wc - shape_wc[i] if stop_wc > shape_wc[i] else 0 for i, stop_wc in enumerate(
                stops_wc))
            stops_wc = [min(shape_wc[i], stop_wc) for i, stop_wc in enumerate(stops_wc)]

        data = io.read(starts_wc, stops_wc)

        # pad if necessary
        if pad_left_wc is not None or pad_right_wc is not None:
            pad_left_wc = (0, 0, 0) if pad_left_wc is None else pad_left_wc
            pad_right_wc = (0, 0, 0) if pad_right_wc is None else pad_right_wc
            assert all(pad_right_wc % res == 0 for pad_right_wc, res in zip(pad_right_wc, io.voxel_size))
            assert all(pad_left_wc % res == 0 for pad_left_wc, res in zip(pad_left_wc, io.voxel_size))
            pad_right_vc = tuple(pad_right_wc / res for pad_right_wc, res in zip(pad_right_wc, io.voxel_size))
            pad_left_vc = tuple(pad_left_wc / res for pad_left_wc, res in zip(pad_left_wc, io.voxel_size))
            pad_width_vc = tuple((pl_vc, pr_vc) for pl_vc, pr_vc in zip(pad_left_vc, pad_right_vc))
            datas.append(np.pad(data, pad_width_vc, mode=padding_mode))
        else:
            datas.append(data)
    return datas



    return datas


def run_inference_n5_multi(prediction,
                           preprocess,
                           postprocess,
                           raw_path,
                           save_file,
                           offset_list,
                           input_shapes_wc,
                           output_shape_wc,
                           input_keys,
                           target_keys,
                           input_resolutions,
                           target_resolutions,
                           padding_mode='reflect',
                           num_cpus=5,
                           log_processed=None,
                           channel_order=None):
    if isinstance(raw_path, str):
        raw_path = [raw_path, ] * len(input_keys)
    if isinstance(save_file, str):
        save_file = [save_file, ] * len(target_keys)
    for rp in raw_path:
        assert os.path.exits(rp)
    for sf in save_file:
        assert os.path.exists(sf)
    assert len(input_keys) == len(raw_path)
    assert len(input_keys) == len(input_resolutions)
    if isinstance(target_keys, str):
        target_keys = (target_keys, )
    if isinstance(input_keys, str):
        input_keys = (input_keys, )

    io_ins = []
    for rp, input_key, input_res in zip(raw_path, input_keys, input_resolutions):
        io_ins.append(IoN5(rp, input_key, input_res))
    io_outs = []
    for sf, target_key, target_res in zip(save_file, target_keys, target_resolutions):
        io_outs.append(IoN5(sf, target_key, target_res))
    #io_out = IoN5(save_file, target_keys, channel_order=channel_order)
    run_inference(prediction, preprocess, postprocess, io_ins, io_outs, offset_list, input_shapes_wc, output_shape_wc,
                  padding_mode=padding_mode, num_cpus=num_cpus, log_processed=log_processed)

def run_inference_n5(prediction,
                     preprocess,
                     postprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_shape_wc,
                     output_shape_wc,
                     input_key,
                     target_keys,
                     input_resolution,
                     target_resolution,
                     padding_mode='reflect',
                     num_cpus=5,
                     log_processed=None,
                     channel_order=None):

    assert os.path.exists(raw_path)
    assert os.path.exists(save_file)
    if isinstance(target_keys, str):
        target_keys = (target_keys,)
    # The N5 IO/Wrapper needs iterables as keys
    # so we wrap the input key in a list.
    # Note that this is not the case for the hdf5 wrapper,
    # which just takes a single key.
    io_in = IoN5(raw_path, input_key, voxel_size=input_resolution)
    io_outs = []
    for target_key in target_keys:
        io_outs.append(IoN5(save_file, target_key, voxel_size=target_resolution))
    run_inference(prediction, preprocess, postprocess, io_in, io_outs,
                  offset_list, input_shape_wc, output_shape_wc, padding_mode=padding_mode,
                  num_cpus=num_cpus, log_processed=log_processed)
    # This is not necessary for n5 datasets
    # which do not need to be closed, but we leave it here for
    # reference when using other (hdf5) io wrappers
    io_in.close()
    for io_out in io_outs:
        io_out.close()


def run_inference_h5(prediction,
                     preprocess,
                     postprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_shape_wc,
                     output_shape_wc,
                     input_key,
                     target_keys,
                     padding_mode='reflect',
                     num_cpus=5,
                     log_processed=None,
                     channel_order=None):

    assert os.path.exists(raw_path)
    assert os.path.exists(raw_path)
    assert os.path.exists(save_file)
    if isinstance(target_keys, str):
        target_keys = (target_keys,)
    # The N5 IO/Wrapper needs iterables as keys
    # so we wrap the input key in a list.
    # Note that this is not the case for the hdf5 wrapper,
    # which just takes a single key.
    io_in = IoHDF5(raw_path, input_key)
    io_outs = []
    for target_key in target_keys:
        io_outs.append(IoHDF5(save_file, target_keys))
    run_inference(prediction, preprocess, postprocess, io_in, io_outs,
                  offset_list, input_shape_wc, output_shape_wc, padding_mode=padding_mode,
                  num_cpus=num_cpus, log_processed=log_processed)
    # This is not necessary for n5 datasets
    # which do not need to be closed, but we leave it here for
    # reference when using other (hdf5) io wrappers
    io_in.close()
    for io_out in io_outs:
        io_out.close()


def run_inference(prediction,
                  preprocess,
                  postprocess,
                  io_ins,
                  io_outs,
                  offset_list,
                  input_shapes_wc,
                  output_shape_wc,
                  padding_mode='reflect',
                  num_cpus=5,
                  log_processed=None):

    assert callable(prediction)
    assert callable(preprocess)

    n_blocks = len(offset_list)
    print("Starting prediction...")
    print("For %i blocks" % n_blocks)
    if not (isinstance(io_ins, list) or isinstance(io_ins, tuple)):
        io_ins = (io_ins,)
        input_shapes_wc = (input_shapes_wc,)

    # the additional context requested in the input
    contexts_wc = []
    for in_sh_wc in input_shapes_wc:
        context_wc = np.array([in_sh_wc[i] - output_shape_wc[i]
                        for i in range(len(in_sh_wc))]) / 2
        contexts_wc.append(tuple(context_wc.astype('uint32')))

    shape_wc = io_outs[0].shape # assume those are the same
    assert [io_out.shape == shape_wc for io_out in io_outs], "different output shapes is not implemented yet"
    @dask.delayed
    def load_offset(offset_wc):
        return load_input(io_ins, offset_wc, contexts_wc, output_shape_wc,
                          padding_mode=padding_mode)

    preprocess = dask.delayed(preprocess)
    predict = dask.delayed(prediction)

    if postprocess is not None:
        postprocess = dask.delayed(postprocess)

    # @dask.delayed(nout=2)
    # def verify_shape(offset_wc, output):
    #     def verify_array_shape(offset_arr, out_arr, io_outs):
    #         # crop if necessary
    #         if out_arr.ndim == 4:
    #             stops = [off + outs for off, outs in zip(offset_arr, out_arr.shape[1:])]
    #         elif out_arr.ndim == 3:
    #             stops = [off + outs for off, outs in zip(offset_arr, out_arr.shape)]
    #         if any(stop > dim_size for stop, dim_size in zip(stops, shape)):
    #             if out_arr.ndim == 4:
    #                 bb = ((slice(None),) +
    #                       tuple(slice(0, dim_size - off if stop > dim_size else None)
    #                             for stop, dim_size, off in zip(stops, shape, offset_arr)))
    #             elif out_arr.ndim == 3:
    #                 bb = (tuple(slice(0, dim_size - off if stop > dim_size else None)
    #                             for stop, dim_size, off in zip(stops, shape, offset_arr)))
    #             out_arr = out_arr[bb]
    #
    #         output_bounding_b = tuple(slice(off, off + outs)
    #                                     for off, outs in zip(offset_arr, output_shape))
    #
    #         return out_arr, output_bounding_b
    #
    #     if isinstance(output, list):
    #         verified_outputs = []
    #         output_bounding_box = []
    #         for out in output:
    #             assert isinstance(out, np.ndarray)
    #             o, bb = verify_array_shape(offset_wc, out)
    #             verified_outputs.append(o)
    #             output_bounding_box.append(bb)
    #         return verified_outputs, output_bounding_box
    #     elif isinstance(output, np.ndarray):
    #         return verify_array_shape(offset, output)
    #     else:
    #         raise TypeError("don't know what to do with output of type"+type(output))

    @dask.delayed(nout=2)
    def verify_shape(offset_wc, output):
        outs = []
        for io_out, out in zip(io_outs, output):
            out = io_out.verify_block_shape(offset_wc, out)
            outs.append(out)
        return outs

    @dask.delayed
    def write_output(output, offsets_wc):
        for io_out, out in zip(io_outs, output):
            io_out.write(out, offsets_wc)
        return 1

    @dask.delayed
    def log(off):
        if log_processed is not None:
            with open(log_processed, 'a') as log_f:
                log_f.write(json.dumps(off) + ', ')
        return off

    # iterate over all the offsets, get the input data and predict
    results = []
    for offsets_wc in offset_list:
        output = tz.pipe(offsets_wc, log, load_offset, preprocess, predict)
        output_crop = verify_shape(offsets_wc, output)
        if postprocess is not None:
            output_crop = postprocess(output_crop, offsets_wc)
        result = write_output(output_crop, offsets_wc)
        results.append(result)

    get = functools.partial(dask.threaded.get, num_workers=num_cpus)
    # NOTE: Because dask.compute doesn't take an argument, but rather an
    # arbitrary number of arguments, computing each in turn, the output of
    # dask.compute(results) is a tuple of length 1, with its only element
    # being the results list. If instead we pass the results list as *args,
    # we get the desired container of results at the end.
    success = dask.compute(*results, get=get)
    print('Ran {0:} jobs'.format(sum(success)))



    @dask.delayed
    def write_output(output, offsets_wc):
        for io_out, out in zip(io_outs, output):
            io_out.write(out, offsets_wc)
        return 1

    @dask.delayed
    def log(off):
        if log_processed is not None:
            with open(log_processed, 'a') as log_f:
                log_f.write(json.dumps(off) + ', ')
        return off

    # iterate over all the offsets, get the input data and predict
    results = []
    for offsets_wc in offset_list:
        output = tz.pipe(offsets_wc, log, load_offset, preprocess, predict)
        output_crop = verify_shape(offsets_wc, output)
        if postprocess is not None:
            output_crop = postprocess(output_crop, offsets_wc)
        result = write_output(output_crop, offsets_wc)
        results.append(result)

    get = functools.partial(dask.threaded.get, num_workers=num_cpus)
    # NOTE: Because dask.compute doesn't take an argument, but rather an
    # arbitrary number of arguments, computing each in turn, the output of
    # dask.compute(results) is a tuple of length 1, with its only element
    # being the results list. If instead we pass the results list as *args,
    # we get the desired container of results at the end.
    success = dask.compute(*results, get=get)
    print('Ran {0:} jobs'.format(sum(success)))
