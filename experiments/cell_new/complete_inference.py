from __future__ import print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call
sys.path.append('/groups/saalfeld/home/heinrichl/tmp/simpleference')
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')
from simpleference.inference.util import get_offset_lists
#from simpleference.inference.util import offset_list_from_precomputed
import z5py
import numpy as np

#from precompute_offsets import precompute_offset_list


def single_inference(gpu, iteration):
    call(['./run_inference.sh', str(gpu),
          str(iteration)])
    return True


def complete_inference(gpu_list, iteration):
    path = '/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5'
    #path = '/groups/saalfeld/saalfeldlab/larissa/data/cell/test_cell2_v1.n5'
    assert os.path.exists(path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(path, use_zarr_format=False)
    assert 'volumes/raw' in rf, "Raw data not present in N5 dataset"
    #assert 'volumes/orig_raw' in rf, "Raw data not present in N5 dataset"
    shape = rf['volumes/raw'].shape
    #shape = rf['volumes/orig_raw'].shape

    # create the datasets
    output_shape = (236, 236, 236)
    out_file = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_{0:}.n5'.format(iteration)
    #out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test_cell2_v1_{0:}.n5'.format(iteration)
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    f = z5py.File(out_file, use_zarr_format=False)
    # the n5 datasets might exist already

    f.create_dataset('cell',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('plasma_membrane',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('ERES',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('ERES_membrane',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('MVB',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('MVB_membrane',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('er',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('er_membrane',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('mito',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('mito_membrane',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('vesicles',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('vesicles_membrane',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)
    f.create_dataset('microtubules',
                     shape=shape,
                     compression='gzip',
                     dtype='uint8',
                     chunks=output_shape)

    # make the offset files, that assign blocks to gpus
    # generate offset lists with mask
    get_offset_lists(shape, gpu_list, out_file, output_shape=output_shape)
    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, gpu, iteration) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")

if __name__ == '__main__':
    gpu_list = [0,2,3,6,7]
    iteration = 104000
    complete_inference(gpu_list, iteration)
