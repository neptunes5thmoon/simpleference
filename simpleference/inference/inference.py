from __future__ import print_function
import os
import json

import numpy as np
import dask
import toolz as tz
import functools

from .io import IoN5, IoHDF5  # IoDVID


def load_input(ios, offset, contexts, output_shape, padding_mode='reflect'):
    if isinstance(ios, tuple) or isinstance(ios, list):
        assert (isinstance(contexts[0], tuple) or isinstance(contexts[0], list))
        assert len(contexts) == len(ios)
    else:
        ios = (ios, )
        contexts = (contexts, )
    datas  = []
    for io, context in zip(ios, contexts):
        starts = [off - context[i] for i, off in enumerate(offset)]
        stops = [off + output_shape[i] + context[i] for i, off in enumerate(offset)]
        shape = io.shape

        # we pad the input volume if necessary
        pad_left = None
        pad_right = None

        # check for padding to the left
        if any(start < 0 for start in starts):
            pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
            starts = [max(0, start) for start in starts]

        # check for padding to the right
        if any(stop > shape[i] for i, stop in enumerate(stops)):
            pad_right = tuple(stop - shape[i] if stop > shape[i] else 0 for i, stop in enumerate(stops))
            stops = [min(shape[i], stop) for i, stop in enumerate(stops)]

        bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
        data = io.read(bb)

        # pad if necessary
        if pad_left is not None or pad_right is not None:
            pad_left = (0, 0, 0) if pad_left is None else pad_left
            pad_right = (0, 0, 0) if pad_right is None else pad_right
            pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
            datas.append(np.pad(data, pad_width, mode=padding_mode))

    return datas


def run_inference_n5_multi(prediction,
                          preprocess,
                          postprocess,
                          raw_path,
                          save_file,
                          offset_list,
                          input_shapes,
                          output_shape,
                          input_keys,
                          target_keys,
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
    assert len(input_keys) == len(raw_path)
    assert os.path.exists(save_file)
    if isinstance(target_keys, str):
        target_keys = (target_keys, )
    if isinstance(input_keys, str):
        input_keys = (input_keys, )

    io_ins = []
    for rp, input_key in zip(raw_path, input_keys):
        io_ins.append(IoN5(rp, input_key))
    io_outs = []
    for sf, target_key in zip(save_file, target_keys):
        io_outs.append(IoN5(sf, target_key))
    #io_out = IoN5(save_file, target_keys, channel_order=channel_order)
    run_inference(prediction, preprocess, postprocess, io_ins, io_outs, offset_list, input_shapes, output_shape,
                  padding_mode=padding_mode, num_cpus=num_cpus, log_processed=log_processed)

def run_inference_n5(prediction,
                     preprocess,
                     postprocess,
                     raw_path,
                     save_file,
                     offset_list,
                     input_shape,
                     output_shape,
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
    io_in = IoN5(raw_path, [input_key])
    io_outs = []
    for target_key in target_keys:
        io_outs.append(IoN5(save_file,target_key))
    run_inference(prediction, preprocess, postprocess, io_in, io_outs,
                  offset_list, input_shape, output_shape, padding_mode=padding_mode,
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
                     input_shape,
                     output_shape,
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
    io_in = IoHDF5(raw_path, [input_key])

    io_out = IoHDF5(save_file, target_keys, channel_order=channel_order)
    run_inference(prediction, preprocess, postprocess, io_in, io_out,
                  offset_list, input_shape, output_shape, padding_mode=padding_mode,
                  num_cpus=num_cpus, log_processed=log_processed)
    # This is not necessary for n5 datasets
    # which do not need to be closed, but we leave it here for
    # reference when using other (hdf5) io wrappers
    io_in.close()
    io_out.close()


def run_inference(prediction,
                  preprocess,
                  postprocess,
                  io_ins,
                  io_outs,
                  offset_list,
                  input_shapes,
                  output_shape,
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
        input_shapes = (input_shapes,)

    # the additional context requested in the input
    contexts = []
    for in_sh in input_shapes:
        context = np.array([in_sh[i] - output_shape[i]
                        for i in range(len(in_sh))]) / 2
        contexts.append(context.astype('uint32'))

    shape = io_outs[0].shape # assume those are the same
    assert [io_out.shape == shape for io_out in io_outs], "different output shapes is not implemented yet"
    @dask.delayed
    def load_offset(offset):
        return load_input(io_ins, offset, contexts, output_shape,
                          padding_mode=padding_mode)

    preprocess = dask.delayed(preprocess)
    predict = dask.delayed(prediction)

    if postprocess is not None:
        postprocess = dask.delayed(postprocess)

    @dask.delayed(nout=2)
    def verify_shape(offset, output):
        def verify_array_shape(offset_arr, out_arr):
            # crop if necessary
            if out_arr.ndim == 4:
                stops = [off + outs for off, outs in zip(offset_arr, out_arr.shape[1:])]
            elif out_arr.ndim == 3:
                stops = [off + outs for off, outs in zip(offset_arr, out_arr.shape)]
            if any(stop > dim_size for stop, dim_size in zip(stops, shape)):
                if out_arr.ndim == 4:
                    bb = ((slice(None),) +
                          tuple(slice(0, dim_size - off if stop > dim_size else None)
                                for stop, dim_size, off in zip(stops, shape, offset_arr)))
                elif out_arr.ndim == 3:
                    bb = (tuple(slice(0, dim_size - off if stop > dim_size else None)
                                for stop, dim_size, off in zip(stops, shape, offset_arr)))
                out_arr = out_arr[bb]

            output_bounding_b = tuple(slice(off, off + outs)
                                        for off, outs in zip(offset_arr, output_shape))

            return out_arr, output_bounding_b

        if isinstance(output, list):
            verified_outputs = []
            output_bounding_box = []
            for out in output:
                assert isinstance(out, np.ndarray)
                o, bb = verify_array_shape(offset, out)
                verified_outputs.append(o)
                output_bounding_box.append(bb)
            return verified_outputs, output_bounding_box
        elif isinstance(output, np.ndarray):
            return verify_array_shape(offset, output)
        else:
            raise TypeError("don't know what to do with output of type"+type(output))

    @dask.delayed
    def write_output(output, output_bounding_box):
        for io_out, out in zip(io_outs, output):
            io_out.write(output, output_bounding_box)
        return 1

    @dask.delayed
    def log(off):
        if log_processed is not None:
            with open(log_processed, 'a') as log_f:
                log_f.write(json.dumps(off) + ', ')
        return off

    # iterate over all the offsets, get the input data and predict
    results = []
    for offset in offset_list:
        output = tz.pipe(offset, log, load_offset, preprocess, predict)
        output_crop, output_bounding_box = verify_shape(offset, output)
        if postprocess is not None:
            output_crop = postprocess(output_crop, output_bounding_box)
        result = write_output(output_crop, output_bounding_box)
        results.append(result)

    get = functools.partial(dask.threaded.get, num_workers=num_cpus)
    # NOTE: Because dask.compute doesn't take an argument, but rather an
    # arbitrary number of arguments, computing each in turn, the output of
    # dask.compute(results) is a tuple of length 1, with its only element
    # being the results list. If instead we pass the results list as *args,
    # we get the desired container of results at the end.
    success = dask.compute(*results, get=get)
    print('Ran {0:} jobs'.format(sum(success)))
