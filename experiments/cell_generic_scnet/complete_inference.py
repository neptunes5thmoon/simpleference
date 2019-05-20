from __future__ import print_function
import sys
import os
import json
from concurrent.futures import ProcessPoolExecutor

from subprocess import call
sys.path.append('/groups/saalfeld/home/heinrichl/construction/simpleference')
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/CNNectome')
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/git_repos/gunpowder')
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')
from simpleference.inference.util import get_offset_lists, redistribute_offset_lists
from utils.label import Label
#from simpleference.inference.util import offset_list_from_precomputed
import z5py
import numpy as np

#from precompute_offsets import precompute_offset_list


def single_inference(gpu, iteration, path, out_file, scale, shift):
    call(['./run_inference.sh', str(gpu),
          str(iteration), str(path), str(out_file), str(scale), str(shift)])
    return True

def vc_to_wc(vc, vs):
    return tuple(np.array(vc)* np.array(vs))

def complete_inference(path, min_sc, max_sc, out_file, gpu_list, iteration, labels, compute_offset_lists=True):
    #path = '/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5'
    #path = '/groups/saalfeld/saalfeldlab/larissa/data/cell/test_cell2_v1.n5'
    assert os.path.exists(path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(path, use_zarr_format=False)
    assert 'volumes/raw/data/s0' in rf, "Raw data not present in N5 dataset"
    assert 'volumes/raw/data/s1' in rf, "Raw data not present in N5 dataset"

    #assert 'volumes/orig_raw' in rf, "Raw data not present in N5 dataset"
    #shape = rf['volumes/orig_raw'].shape

    # create the datasets

    # network_output_shape_vc = (56, 56, 56)
    # chunk_shape_vc = (45, 45, 45)
    # input_shape_0_vc = (124,124,124)
    # input_shape_1_vc = (88,88,88)


    network_output_shape_vc = (173,173,173)
    chunk_shape_vc = (162,162,162)
    input_shape_0_vc = (241,241,241)
    input_shape_1_vc = (97,97,97)
    output_vs = (4,4,4)
    input_vs_0 = (4,4,4)
    input_vs_1 = (36,36,36)
    chunk_vs = (4,4,4)
    chunk_shape_wc = vc_to_wc(chunk_shape_vc, chunk_vs)
    network_output_shape_wc = vc_to_wc(network_output_shape_vc, output_vs)
    input_shape_0_wc = vc_to_wc(input_shape_0_vc, input_vs_0)
    input_shape_1_wc = vc_to_wc(input_shape_1_vc, input_vs_1)
    shape_vc = rf['volumes/raw/data/s0'].shape
    shape_wc=vc_to_wc(shape_vc, input_vs_0)

    #out_file = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_{0:}.n5'.format(iteration)
    #out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test_cell2_v1_{0:}.n5'.format(iteration)
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    f = z5py.File(out_file, use_zarr_format=False)
    # the n5 datasets might exist already
    if compute_offset_lists:
        for label in labels:
            f.require_dataset(label.labelname, shape=shape_vc, compression='gzip', dtype='uint8', chunks=chunk_shape_vc)
    # make the offset files, that assign blocks to gpus
    # generate offset lists with mask
    if compute_offset_lists:
        get_offset_lists(shape_wc, gpu_list, out_file, output_shape=chunk_shape_wc)
    else:
        redistribute_offset_lists(gpu_list, out_file)
    # run multiprocessed inference
    #scale = 1./(max_sc - min_sc)
    #shift = - min_sc * scale
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
        tasks = [pp.submit(single_inference, gpu, iteration, path, out_file, min_sc, max_sc) for gpu in gpu_list]
        result = [t.result() for t in tasks]

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")
    re_gpu_list = []
    for gpu in gpu_list:
        if not check_completeness(out_file, gpu):
            re_gpu_list.append(gpu)
    if len(re_gpu_list) > 0:
        complete_inference(path, min_sc, max_sc, out_file, re_gpu_list, iteration, labels, compute_offset_lists=False)


def check_completeness(out_file, gpu):
    if os.path.exists(os.path.join(out_file, 'list_gpu_{0:}.json'.format(gpu))) and os.path.exists(
            os.path.join(out_file, 'list_gpu_{0:}_{1:}_processed.txt'.format(gpu, iteration))):
        block_list = os.path.join(out_file, 'list_gpu_{0:}.json'.format(gpu))
        block_list_processed = os.path.join(out_file, 'list_gpu_{0:}_{1:}_processed.txt'.format(gpu, iteration))
        with open(block_list, 'r') as f:
            block_list = json.load(f)
            block_list = {tuple(coo) for coo in block_list}
        with open(block_list_processed, 'r') as f:
            list_as_str = f.read()
        list_as_str_curated = '[' + list_as_str[:list_as_str.rfind(']') + 1] + ']'
        processed_list = json.loads(list_as_str_curated)
        processed_list = {tuple(coo) for coo in processed_list}
        if processed_list < block_list:
            complete = False
        else:
            complete = True
    else:
        complete = False
    return complete


if __name__ == '__main__':
    gpu_list = [2,3,4,5,6,7]
    iteration = 535000
    path = '/groups/saalfeld/saalfeldlab/larissa/data/cell/test-data/test2.n5'
    out_path = '/nrs/saalfeld/heinrichl/cell/scalenet/01-030319/run01-restart/test2_{0:}.n5'.format(iteration)
    min_sc = 70.
    max_sc = 204.
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/TWalther_WT45_Cell2_4x4x4nm' \
    #       '/Cryo_20171009_WT45_Cell2_4x4x4nm.n5'
    #out_path = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/walther_{0:}.n5'.format(iteration)
    #min_sc = 0.57
    #max_sc = 0.99
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/Chlamydomonas' \
    #        '/Chlamydomonas_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/chlamydomonas_{0:}.n5'.format(iteration)
    # min_sc = 0.3
    # max_sc=1.
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/Macrophage_FS80_Cell2_4x4x4nm' \
    #         '/Cryo_FS80_Cell2_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/macrophage_{0:}.n5'.format(iteration)
    # min_sc = 0.71
    # max_sc=1.
    #BEFORE STARTING A NEW PREDICTION CHANGE SCALING
    #path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/HeLa_Cell3_4x4x4nm/Aubrey_17-7_17_Cell3%204x4x4nm.n5'
    #out_path = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/cell3_{0:}.n5'.format(iteration)
    #min_sc = -0.55
    #max_sc=1.64

    labels = []
    labels.append(Label('ecs', 1, ))
    labels.append(Label('plasma_membrane', 2, ))
    labels.append(Label('mito', (3, 4, 5), ))
    labels.append(Label('mito_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key,
                        ))
    labels.append(Label('mito_DNA', 5, scale_loss=False, scale_key=labels[-2].scale_key, ))
    labels.append(Label('golgi', (6, 7), ))
    labels.append(Label('golgi_membrane', 6, ))
    labels.append(Label('vesicle', (8, 9), ))
    labels.append(Label('vesicle_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key,
                        ))
    labels.append(Label('MVB', (10, 11), ))
    labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key, ))
    labels.append(Label('lysosome', (12, 13), ))
    labels.append(Label('lysosome_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key,
                        ))
    labels.append(Label('LD', (14, 15), ))
    labels.append(Label('LD_membrane', 14, scale_loss=False, scale_key=labels[-1].scale_key, ))
    labels.append(Label('er', (16, 17, 18, 19, 20, 21, 22, 23), ))
    labels.append(Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key,
                        ))
    labels.append(Label('ERES', (18, 19), ))
    #labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    ))
    labels.append(Label('nucleus', (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36),
                        ))
    labels.append(Label('nucleolus', 29, ))
    labels.append(Label('NE', (20, 21, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key,
                        ))
    #labels.append(Label('NE_membrane', (20, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key,
    # ))
    labels.append(Label('nuclear_pore', (22, 23), ))
    labels.append(Label('nuclear_pore_out', 22, scale_loss=False, scale_key=labels[-1].scale_key,
                        ))
    labels.append(Label('chromatin', (24, 25, 26, 27, 36), ))
    #labels.append(Label('NHChrom', 25, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources,
    # ))
    #labels.append(Label('EChrom', 26, scale_loss=False, scale_key=labels[-2].scale_key, data_sources=data_sources,
    # ))
    #labels.append(Label('NEChrom', 27, scale_loss=False, scale_key=labels[-3].scale_key, data_sources=data_sources,
    # ))
    labels.append(Label('NHChrom', 25, ))
    labels.append(Label('EChrom', 26, ))
    labels.append(Label('NEChrom', 27, ))
    labels.append(Label('microtubules', (30, 31), ))
    labels.append(Label('centrosome', (31, 32, 33), ))
    labels.append(Label('distal_app', 32, ))
    labels.append(Label('subdistal_app', 33, ))
    labels.append(Label('ribosomes', 1 ))
    complete_inference(path, min_sc, max_sc, out_path, gpu_list, iteration, labels, compute_offset_lists=True)
