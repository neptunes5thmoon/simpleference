from __future__ import print_function
import sys
import os
import json
from concurrent.futures import ProcessPoolExecutor

from subprocess import call
sys.path.append('/groups/saalfeld/home/heinrichl/construction/simpleference')
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/CNNectome')
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/git_repos/gunpowder')
#sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')
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


def complete_inference(path, min_sc, max_sc, out_file, gpu_list, iteration, labels, compute_offset_lists=True):
    assert os.path.exists(path), "Path to N5 dataset with raw data and mask does not exist"
    rf = z5py.File(path, use_zarr_format=False)
    assert 'volumes/raw' in rf, "Raw data not present in N5 dataset"
    shape_vc = rf['volumes/raw'].shape

    # create the datasets
    # output_shape_vc = (198,198,198)
    # chunk_shape_vc = (198,198,198)
    # output_shape_vc = (104, 104, 104)
    # chunk_shape_vc = (104, 104, 104)
    output_shape_vc = (204, 204, 204)
    chunk_shape_vc = (204, 204, 204)
    resolution = (4, 4, 4)
    input_resolution = (8, 8, 8)
    output_shape_wc = tuple(np.array(output_shape_vc)* np.array(resolution))
    chunk_shape_wc = tuple(np.array(chunk_shape_vc)* np.array(resolution))
    shape_wc = tuple(np.array(shape_vc) * np.array(input_resolution))
    output_shape_vc = tuple(np.array(shape_wc)/np.array(resolution))
    #out_file = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_{0:}.n5'.format(iteration)
    #out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test_cell2_v1_{0:}.n5'.format(iteration)
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    f = z5py.File(out_file, use_zarr_format=False)
    # the n5 datasets might exist already
    if compute_offset_lists:
        for label in labels:
            f.require_dataset(label.labelname, shape=output_shape_vc, compression='gzip', dtype='uint8',
                              chunks=chunk_shape_vc)
    # make the offset files, that assign blocks to gpus
    # generate offset lists with mask
    if compute_offset_lists:
        get_offset_lists(shape_wc, gpu_list, out_file, chunk_shape_wc)
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
    gpu_list = [5,6,7]
    iteration = 141500
    # path = '/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/gt061719/8to4/unet/02-070519/test2_{0:}_89_207.n5'.format(iteration)
    # path = '/groups/cosem/cosem/data/HeLa_Cell2_4x4x4nm/Aubrey_17' \
    #        '-7_17_Cell2_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/unet/01-030819/cell2_{0:}_89_207.n5'.format(iteration)
    # min_sc = 70.
    # max_sc = 204.
    # min_sc = 89.
    # max_sc = 207.
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/8x8x8nm_Data/Cell21_FIB.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/unet/01-030819/cell21_8nm.n5'
    # min_sc = 255 * .09
    # max_sc = 255 * .95
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/Jurkat_Cell1_4x4x4nm/Jurkat_Cell1_FS96-Area1_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/unet/01-030819/jurkat_cell1_{0:}.n5'.format(iteration)
    # min_sc = 255*.792
    # max_sc = 255*.934
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/Pancreas_Islets/Pancreas_G36-2_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/unet/01-030819/pancreas_{0:}.n5'.format(iteration)
    # min_sc = 255 *.667
    # max_sc = 255 *.780
    path = '/groups/cosem/cosem/data/COS7_Cell11_8x8x8nm' \
            '/Cryo_LoadID277_Cell11_8x8x8nm_bigwarped_v17.n5'
    out_path = '/nrs/saalfeld/heinrichl/cell/gt061719/8to4/unet/02-070519/COS7_cell11_{0:}.n5'.format(iteration)
    min_sc = 55.
    max_sc = 186.
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/HeLa_Cell3_4x4x4nm/Aubrey_17-7_17_Cell3_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/unet/01-030819/hela_cell3_{0:}.n5'.format(iteration)
    # min_sc = 255 * 0.
    # max_sc = 255 * 1.1


    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/TWalther_WT45_Cell2_4x4x4nm' \
    #       '/Cryo_20171009_WT45_Cell2_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/unet/01-030819/SUM159_walther_{0:}.n5'.format(iteration)
    # min_sc = 172.
    # max_sc = 233.
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/Chlamydomonas' \
    #        '/Chlamydomonas_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/chlamydomonas_{0:}.n5'.format(iteration)
    # min_sc = 0.3
    # max_sc=1.
    # path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/Macrophage_FS80_Cell2_4x4x4nm' \
    #         '/Cryo_FS80_Cell2_4x4x4nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/unet/01-030819/macrophage_{0:}.n5'.format(iteration)
    # min_sc = 0.75*255
    # max_sc = 255*1.

    # path = '/groups/cosem/cosem/data/HeLa_Cell21_8x8x8nm/HeLa_Cell21_8x8x8nm.n5'
    # out_path = '/nrs/saalfeld/heinrichl/cell/gt061719/8to4/unet/02-070519/hela_cell21_{0:}.n5'.format(
    #     iteration)
    # min_sc = 0.
    # max_sc = 255.

    #BEFORE STARTING A NEW PREDICTION CHANGE SCALING
    #path = '/groups/hess/hess_collaborators/Annotations/ParentFiles_whole-cell_images/HeLa_Cell3_4x4x4nm/Aubrey_17-7_17_Cell3%204x4x4nm.n5'
    #out_path = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/cell3_{0:}.n5'.format(iteration)
    #min_sc = -0.55
    #max_sc=1.64

    data_dir = '{0:}'
    data_sources = []
    ribo_sources = []
    nucleolus_sources = []
    centrosomes_sources = []

    labels = list()
    labels.append(Label('ecs', 1, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('plasma_membrane', 2, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('mito', (3, 4, 5), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('mito_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('mito_DNA', 5, scale_loss=False, scale_key=labels[-2].scale_key, data_sources=data_sources,
    #                    data_dir=data_dir))
    # labels.append(Label('golgi', (6, 7), data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('golgi_membrane', 6, data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('vesicle', (8, 9), data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('vesicle_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('MVB', (10, 11), data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources,
    #                    data_dir=data_dir))
    # labels.append(Label('lysosome', (12, 13), data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('lysosome_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('LD', (14, 15), data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('LD_membrane', 14, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources,
    #                    data_dir=data_dir))
    labels.append(Label('er', (16, 17, 18, 19, 20, 21, 22, 23), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('ERES', (18, 19), data_sources=data_sources, data_dir=data_dir))
    # labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('nucleus', (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36), data_sources=data_sources,
                        data_dir=data_dir))
    complete_inference(path, min_sc, max_sc, out_path, gpu_list, iteration, labels, compute_offset_lists=True)
