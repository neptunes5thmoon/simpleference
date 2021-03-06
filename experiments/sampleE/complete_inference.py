from __future__ import print_function
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from subprocess import call
from simpleference.inference.util import get_offset_lists

sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld/python')
import z5py


def single_inference(gpu, iteration, gpu_offset):
    print(gpu, iteration, gpu_offset)
    call(['./run_inference.sh', str(gpu), str(iteration), str(gpu_offset)])
    return True


def complete_inference(gpu_list, iteration, gpu_offset):

    out_shape = (56,) *3

    raw_path = '/nrs/saalfeld/sample_E/sample_E.n5/volumes/raw/s0'
    g = z5py.File(raw_path)
    shape = g['.'].shape[::-1]

    # open the datasets
    #f = z5py.File('/groups/saalfeld/saalfeldlab/sampleE/my_prediction.n5', use_zarr_format=False)
    #f.create_dataset('affs_xy', shape=shape,
    #                 compression='gzip',
    #                 dtype='float32',
    #                 chunks=out_shape)
    #f.create_dataset('affs_z', shape=shape,
    #                 compression='gzip',
    #                 dtype='float32',
    #                 chunks=out_shape)

    # run multiprocessed inference
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, gpu, iteration, gpu_offset) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu_list = range(8)
    iteration = 100000
    gpu_offset = int(sys.argv[1])
    complete_inference(gpu_list, iteration, gpu_offset)
