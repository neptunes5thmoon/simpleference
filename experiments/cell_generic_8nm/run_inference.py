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


def type_conversion(data):
    return [d.astype(np.uint8) for d in data]


def clip_preprocess(data, scale=2, shift=-1):
    return clip(scale_shift(normalize(data), scale, shift))


def both(data, scale=2, shift=-1):
    return clip_preprocess(type_conversion(data), scale=scale, shift=shift)


def single_gpu_inference(gpu, iteration, labels, path, out_file, min_sc=0., max_sc=255.):

    # path = '/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5'
    # path = '/groups/saalfeld/saalfeldlab/larissa/data/cell/test_cell2_v1.n5'
    assert os.path.exists(path), path
    rf = z5py.File(path, use_zarr_format=False)
    shape = rf['volumes/raw'].shape
    #shape = rf['volumes/orig_raw'].shape
    weight_meta_graph = '/nrs/saalfeld/heinrichl/cell/gt061719/8to4/unet/02-070519/unet_train_checkpoint_%i' % iteration
    inference_meta_graph = '/nrs/saalfeld/heinrichl/cell/gt061719/8to4/unet/02-070519/unet_inference'
    net_io_json = '/nrs/saalfeld/heinrichl/cell/gt061719/8to4/unet/02-070519/net_io_names.json'

    # out_file ='/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_{0:}.n5'.format(iteration)
    # out_file ='/nrs/saalfeld/heinrichl/cell/gt110618/setup03/run01/test_cell2_v1_{0:}.n5'.format(iteration)
    with open(net_io_json, 'r') as f:
        net_io_names = json.load(f)

    offset_file = out_file+'/list_gpu_{0:}.json'.format(gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)
    # offset_list = [[150*4, 150*4, 150*4]]
    input_key = net_io_names["raw"]
    network_output_keys = []
    dataset_target_keys = []
    for label in labels:
        network_output_keys.append(net_io_names[label.labelname])
        dataset_target_keys.append(label.labelname)

    # chunk_shape_vc = (198, 198, 198)
    # input_shape_vc = (342, 342, 342)
    # output_shape_vc = (198, 198, 198)
    # chunk_shape_vc = (200, 200, 200)
    # input_shape_vc = (304, 304, 304)
    # output_shape_vc = (200, 200, 200)

    # chunk_shape_vc = (104, 104, 104)
    # output_shape_vc = (104, 104, 104)
    # input_shape_vc = (208, 208, 208)
    chunk_shape_vc = (204, 204, 204)
    output_shape_vc = (204, 204, 204)
    input_shape_vc = (208, 208, 208)
    output_resolution = (4, 4, 4)
    input_resolution = (8, 8, 8)
    output_shape_wc = tuple(np.array(output_shape_vc) * np.array(output_resolution))
    input_shape_wc = tuple(np.array(input_shape_vc) * np.array(input_resolution))
    chunk_shape_wc = tuple(np.array(chunk_shape_vc) * np.array(output_resolution))
    prediction = TensorflowPredict(weight_meta_graph,
                                   inference_meta_graph,
                                   input_keys=input_key,
                                   output_keys=network_output_keys)
    t_predict = time.time()

    scale = 255. / (float(max_sc) - float(min_sc))
    shift = -scale * (float(min_sc) / 255.)
    run_inference_n5_multi_crop(prediction,
                     partial(both, scale=scale*2., shift=shift*2.-1.),
                     partial(clip_float_to_uint8, float_range=(-1., 1.), safe_scale=False),
                     path,
                     out_file,
                     offset_list,
                     network_input_shapes_wc=[input_shape_wc, ],
                     network_output_shape_wc=output_shape_wc,
                     chunk_shape_wc=chunk_shape_wc,
                     input_keys=['volumes/raw', ],
                     target_keys=dataset_target_keys,
                     input_resolutions=[input_resolution, ],
                     target_resolutions=[output_resolution, ] * len(dataset_target_keys),
                     log_processed=os.path.join(os.path.dirname(offset_file),
                                                'list_gpu_{0:}_{1:}_processed.txt'.format(gpu, iteration)),
                     pad_value=int(round(-255.*((shift*2.-1.)/(scale*2.))))
                     )

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
    # labels = []
    # labels.append(Label('ecs', 1))
    # labels.append(Label('plasma_membrane', 2))
    # labels.append(Label('mito', (3, 4, 5)))
    # labels.append(Label('mito_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('mito_DNA', 5, scale_loss=False, scale_key=labels[-2].scale_key))
    # labels.append(Label('vesicle', (8, 9)))
    # labels.append(Label('vesicle_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('MVB', (10, 11)))
    # labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('lysosome', (12, 13)))
    # labels.append(Label('lysosome_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('LD', (14, 15)))
    # labels.append(Label('LD_membrane', 14, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('er', (16, 17, 18, 19, 20, 21)))
    # labels.append(Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('ERES', (18, 19)))
    # labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('nucleus', (20,21,24,25)))
    # labels.append(Label('NE', (20, 21), scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('NE_membrane', 20, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('chromatin', 24, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('nucleoplasm', 25, scale_loss=False, scale_key=labels[-1].scale_key))
    # labels.append(Label('microtubules', 26))
    # labels.append(Label('ribosomes', 1))
    # labels = []
    # labels.append(Label('ecs', 1, ))
    # labels.append(Label('plasma_membrane', 2, ))
    # labels.append(Label('mito', (3, 4, 5), ))
    # labels.append(Label('mito_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key,
    #                     ))
    # labels.append(Label('mito_DNA', 5, scale_loss=False, scale_key=labels[-2].scale_key, ))
    # labels.append(Label('golgi', (6, 7), ))
    # labels.append(Label('golgi_membrane', 6, ))
    # labels.append(Label('vesicle', (8, 9), ))
    # labels.append(Label('vesicle_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key,
    #                     ))
    # labels.append(Label('MVB', (10, 11), ))
    # labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key, ))
    # labels.append(Label('lysosome', (12, 13), ))
    # labels.append(Label('lysosome_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key,
    #                     ))
    # labels.append(Label('LD', (14, 15), ))
    # labels.append(Label('LD_membrane', 14, scale_loss=False, scale_key=labels[-1].scale_key, ))
    # labels.append(Label('er', (16, 17, 18, 19, 20, 21, 22, 23), ))
    # labels.append(Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key,
    #                     ))
    # labels.append(Label('ERES', (18, 19), ))
    # #labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key,
    # #                    ))
    # labels.append(Label('nucleus', (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36),
    #                     ))
    # labels.append(Label('nucleolus', 29, ))
    # labels.append(Label('NE', (20, 21, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key,
    #                     ))
    # #labels.append(Label('NE_membrane', (20, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key,
    # # ))
    # labels.append(Label('nuclear_pore', (22, 23), ))
    # labels.append(Label('nuclear_pore_out', 22, scale_loss=False, scale_key=labels[-1].scale_key,
    #                     ))
    # labels.append(Label('chromatin', (24, 25, 26, 27, 36), ))
    # #labels.append(Label('NHChrom', 25, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources,
    # # ))
    # #labels.append(Label('EChrom', 26, scale_loss=False, scale_key=labels[-2].scale_key, data_sources=data_sources,
    # # ))
    # #labels.append(Label('NEChrom', 27, scale_loss=False, scale_key=labels[-3].scale_key, data_sources=data_sources,
    # # ))
    # labels.append(Label('NHChrom', 25, ))
    # labels.append(Label('EChrom', 26, ))
    # labels.append(Label('NEChrom', 27, ))
    # labels.append(Label('microtubules', (30, 31), ))
    # labels.append(Label('centrosome', (31, 32, 33), ))
    # labels.append(Label('distal_app', 32, ))
    # labels.append(Label('subdistal_app', 33, ))
    # labels.append(Label('ribosomes', 1 ))
    # #
    #
    data_sources = []
    data_dir = '{0:}'
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
    single_gpu_inference(gpu, iteration, labels, path, out_file,  scale, shift)
