import os
import sys
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')


import time
import json
import z5py
from functools import partial
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.postprocessing import *

def single_gpu_inference(gpu, iteration):

    path = '/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5'
    #path = '/groups/saalfeld/saalfeldlab/larissa/data/cell/test_cell2_v1.n5'
    assert os.path.exists(path), path
    rf = z5py.File(path, use_zarr_format=False)
    shape = rf['volumes/raw'].shape
    #shape = rf['volumes/orig_raw'].shape
    weight_meta_graph = '/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/unet_checkpoint_%i' % iteration
    inference_meta_graph = '/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/unet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/net_io_names.json'

    out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test2_{0:}.n5'.format(iteration)
    #out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test_cell2_v1_{0:}.n5'.format(iteration)
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = out_file+'/list_gpu_{0:}.json'.format(gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    if os.path.exists(os.path.join(os.path.dirname(offset_file), 'list_gpu_{0:}_{'
                                                                                '1:}_processed.txt'.format(gpu,
                                                                                                           iteration))):
        block_list_processed = os.path.join(out_file, 'list_gpu_{0:}_{1:}_processed.json'.format(gpu, iteration))
        with open(block_list_processed, 'r') as f:
            list_as_str = f.read()
        list_as_str_curated = '[' + list_as_str[:list_as_str.rfind(']') + 1] + ']'
        processed_list = json.loads(list_as_str_curated)
        processed_list = {tuple(coo) for coo in processed_list}
        full_list = {tuple(coo) for coo in offset_list}
        assert processed_list != full_list
        updated_list = full_list-processed_list
        offset_list = [list(coo) for coo in updated_list]

    input_key = net_io_names["raw"]
    output_keys = [net_io_names["cell"],
                   net_io_names["plasma_membrane"],
                   net_io_names["ERES"],
                   net_io_names["ERES_membrane"],
                   net_io_names["MVB"],
                   net_io_names["MVB_membrane"],
                   net_io_names["er"],
                   net_io_names["er_membrane"],
                   net_io_names["mito"],
                   net_io_names["mito_membrane"],
                   net_io_names["vesicles"],
                   net_io_names["vesicles_membrane"],
                   net_io_names["microtubules"],
                   ]
    input_shape = (340, 340, 340)
    output_shape = (236, 236, 236)
    prediction = TensorflowPredict(weight_meta_graph,
                                   inference_meta_graph,
                                   input_key=input_key,
                                   output_key=output_keys)
    t_predict = time.time()

    run_inference_n5(prediction,
                     preprocess,
                     partial(clip_float_to_uint8, float_range=(-1., 1.), safe_scale=True),
                     path,
                     out_file,
                     offset_list,
                     input_shape=input_shape,
                     output_shape=output_shape,
                     target_keys= (
                                   'cell',
                                   'plasma_membrane',
                                   'ERES',
                                   'ERES_membrane',
                                   'MVB',
                                   'MVB_membrane',
                                   'er',
                                   'er_membrane',
                                   'mito',
                                   'mito_membrane',
                                   'vesicles',
                                   'vesicles_membrane',
                                   'microtubules',
                                   ),
                   #  input_key='volumes/orig_raw',
                     input_key='volumes/raw',
                     log_processed=os.path.join(os.path.dirname(offset_file), 'list_gpu_{0:}_{'
                                                                                '1:}_processed.txt'.format(gpu,
                                                                                                           iteration)))

    t_predict = time.time() - t_predict

    with open(os.path.join(os.path.dirname(offset_file), 't-inf_gpu_{0:}_{1:}.txt'.format(gpu, iteration)), 'w') as f:
        f.write("Inference with gpu %i in %f s\n" % (gpu, t_predict))


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    iteration = int(sys.argv[2])
    single_gpu_inference(gpu, iteration)
