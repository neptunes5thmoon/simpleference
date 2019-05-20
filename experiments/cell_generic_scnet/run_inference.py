import os
import sys
sys.path.append('/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python')
sys.path.append('/groups/saalfeld/home/heinrichl/Projects/CNNectome')
import time
import json
import z5py
from functools import partial
from utils.label import Label
from simpleference.inference.inference import run_inference_n5_multi_crop
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import zero_out_const_sections,clip,scale_shift,normalize
from simpleference.postprocessing import *

def clip_preprocess(data, scale=2, shift=-1):
    return clip(scale_shift(normalize(data),scale,shift))

def vc_to_wc(vc, vs):
    return tuple(np.array(vc)* np.array(vs))

def single_gpu_inference(gpu, iteration, labels, path, out_file, min_sc=0., max_sc=255.):

  #  path = '/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5'
    #path = '/groups/saalfeld/saalfeldlab/larissa/data/cell/test_cell2_v1.n5'
    assert os.path.exists(path), path
    rf = z5py.File(path, use_zarr_format=False)
    shape = rf['volumes/raw/data/s0'].shape
    #shape = rf['volumes/orig_raw'].shape
    weight_meta_graph = '/nrs/saalfeld/heinrichl/cell/scalenet/01-030319/run01-restart/scnet_train_checkpoint_%i' % \
                        iteration
    inference_meta_graph = '/nrs/saalfeld/heinrichl/cell/scalenet/01-030319/run01-restart/scnet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/cell/scalenet/01-030319/run01-restart/net_io_names.json'

   # out_file ='/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_{0:}.n5'.format(iteration)
    #out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test_cell2_v1_{0:}.n5'.format(iteration)
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = out_file+'/list_gpu_{0:}.json'.format(gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    #input_key = net_io_names["raw"]
    network_output_keys=[]
    dataset_target_keys=[]
    for label in labels:
        network_output_keys.append(net_io_names[label.labelname])
        dataset_target_keys.append(label.labelname)

    dataset_input_keys = ['volumes/raw/data/s0', 'volumes/raw/data/s2']
    network_input_keys = [net_io_names['raw_4'], net_io_names['raw_36']]

    # network_output_shape_vc = (56, 56, 56)
    # chunk_shape_vc = (45, 45, 45)
    # input_shape_0_vc = (124, 124, 124)
    # input_shape_1_vc = (88, 88, 88)


    network_output_shape_vc = (173,173,173)
    chunk_shape_vc = (162,162,162)
    input_shape_0_vc = (241,241,241)
    input_shape_1_vc = (97,97,97)
    output_vs = (4, 4, 4)
    input_vs_0 = (4, 4, 4)
    input_vs_1 = (36, 36, 36)
    chunk_vs = (4, 4, 4)
    chunk_shape_wc = vc_to_wc(chunk_shape_vc, chunk_vs)
    network_output_shape_wc = vc_to_wc(network_output_shape_vc, output_vs)
    input_shape_0_wc = vc_to_wc(input_shape_0_vc, input_vs_0)
    input_shape_1_wc = vc_to_wc(input_shape_1_vc, input_vs_1)
    prediction = TensorflowPredict(weight_meta_graph,
                                   inference_meta_graph,
                                   input_keys=network_input_keys,
                                   output_keys=network_output_keys)
    t_predict = time.time()

    scale = 255./(float(max_sc)-float(min_sc))
    shift = -scale * (float(min_sc)/255.)
    run_inference_n5_multi_crop(prediction,
                     partial(clip_preprocess, scale=scale*2., shift=shift*2.-1.),
                     partial(clip_float_to_uint8, float_range=(-1., 1.), safe_scale=False),
                     path,
                     out_file,
                     offset_list,
                     network_input_shapes_wc=[input_shape_0_wc, input_shape_1_wc],
                     network_output_shape_wc=network_output_shape_wc,
                     chunk_shape_wc = chunk_shape_wc,
                     input_keys=dataset_input_keys,
                     target_keys=dataset_target_keys,
                     input_resolutions=[input_vs_0, input_vs_1],
                     target_resolutions=[output_vs]*len(dataset_target_keys),
                   #  input_key='volumes/orig_raw',
                     log_processed=os.path.join(os.path.dirname(offset_file), 'list_gpu_{0:}_{'
                                                                                '1:}_processed.txt'.format(gpu,
                                                                                                           iteration)),
                     pad_value = int(round(-255.*((shift*2.-1.)/(scale*2.)))))

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
    labels = list()
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
    single_gpu_inference(gpu, iteration, labels, path, out_file,  scale, shift)
