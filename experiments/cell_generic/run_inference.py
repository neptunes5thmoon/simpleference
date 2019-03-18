import os
import sys
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/CNNectome')
import time
import json
import z5py
from functools import partial
from utils.label import Label
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import zero_out_const_sections,clip,scale_shift,normalize
from simpleference.postprocessing import *

def clip_preprocess(data, scale=2, shift=-1):
    return zero_out_const_sections(clip(scale_shift(normalize(data),
                                                         scale,
                                                         shift)))
def single_gpu_inference(gpu, iteration, labels, path, out_file, scale=2., shift=1.):

  #  path = '/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5'
    #path = '/groups/saalfeld/saalfeldlab/larissa/data/cell/test_cell2_v1.n5'
    assert os.path.exists(path), path
    rf = z5py.File(path, use_zarr_format=False)
    shape = rf['volumes/raw'].shape
    #shape = rf['volumes/orig_raw'].shape
    weight_meta_graph = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/unet_checkpoint_%i' % iteration
    inference_meta_graph = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/unet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/net_io_names.json'

   # out_file ='/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_{0:}.n5'.format(iteration)
    #out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test_cell2_v1_{0:}.n5'.format(iteration)
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = out_file+'/list_gpu_{0:}.json'.format(gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = net_io_names["raw"]
    output_keys=[]
    target_keys=[]
    for label in labels:
        output_keys.append(net_io_names[label.labelname])
        target_keys.append(label.labelname)
    input_shape_vc = (340, 340, 340)
    output_shape_vc = (236, 236, 236)
    resolution = (4, 4, 4)
    output_shape_wc = tuple(np.array(output_shape_vc) * np.array(resolution))
    input_shape_wc = tuple(np.array(input_shape_vc)* np.array(resolution))
    prediction = TensorflowPredict(weight_meta_graph,
                                   inference_meta_graph,
                                   input_keys=input_key,
                                   output_keys=output_keys)
    t_predict = time.time()
    run_inference_n5(prediction,
                     partial(clip_preprocess, scale=scale*2., shift=shift*2.-1.),
                     partial(clip_float_to_uint8, float_range=(-1., 1.), safe_scale=False),
                     path,
                     out_file,
                     offset_list,
                     input_shape_wc=input_shape_wc,
                     output_shape_wc=output_shape_wc,
                     target_keys= target_keys,
                     #  input_key='volumes/orig_raw',
                     input_key='volumes/raw',
                     input_resolution=resolution,
                     target_resolution=resolution,
                     log_processed=os.path.join(os.path.dirname(offset_file), 'list_gpu_{0:}_{'
                                                                                '1:}_processed.txt'.format(gpu,
                                                                                                           iteration)))

    t_predict = time.time() - t_predict

    with open(os.path.join(os.path.dirname(offset_file), 't-inf_gpu_{0:}_{1:}.txt'.format(gpu, iteration)), 'w') as f:
        f.write("Inference with gpu %i in %f s\n" % (gpu, t_predict))


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    iteration = int(sys.argv[2])
    path = str(sys.argv[3])
    out_file = str(sys.argv[4])
    scale = float(sys.argv[5])
    shift = float(sys.argv[6])
    labels = []
    labels.append(Label('ecs', 1))
    labels.append(Label('plasma_membrane', 2))
    labels.append(Label('mito', (3, 4, 5)))
    labels.append(Label('mito_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('mito_DNA', 5, scale_loss=False, scale_key=labels[-2].scale_key))
    labels.append(Label('vesicle', (8, 9)))
    labels.append(Label('vesicle_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('MVB', (10, 11)))
    labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('lysosome', (12, 13)))
    labels.append(Label('lysosome_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('LD', (14, 15)))
    labels.append(Label('LD_membrane', 14, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('er', (16, 17, 18, 19, 20, 21)))
    labels.append(Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('ERES', (18, 19)))
    labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('nucleus', (20,21,24,25)))
    labels.append(Label('NE', (20, 21), scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('NE_membrane', 20, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('chromatin', 24, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('nucleoplasm', 25, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('microtubules', 26))
    labels.append(Label('ribosomes', 1))
    single_gpu_inference(gpu, iteration, labels, path, out_file,  scale, shift)
