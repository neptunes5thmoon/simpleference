import os
import sys
import time
import json
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.caffe.backend import CaffePredict
from simpleference.backends.gunpowder.preprocess import preprocess


def single_gpu_inference(sample, gpu, iteration):
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/n5/cremi_warped_sample%s.n5' % sample
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/gp_caffe_predictions_iter_%i' % iteration
    out_file = os.path.join(out_file, 'cremi_warped_sample%s_predictions_blosc.n5' % sample)
    assert os.path.exists(out_file)

    prototxt = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/long_range_v2/long_range_unet.prototxt'
    weights  = '/groups/saalfeld/home/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/long_range_v2/net_iter_%i.caffemodel' % iteration

    offset_file = './offsets_sample%s/list_gpu_%i.json' % (sample, gpu)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_key = 'data'
    output_key = 'aff_pred'
    input_shape = (84, 268, 268)
    output_shape = (56, 56, 56)
    prediction = CaffePredict(prototxt,
                              weights,
                              input_key=input_key,
                              output_key=output_key,
                              gpu=gpu)
    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     out_file,
                     offset_list,
                     input_shape_wc=input_shape,
                     output_shape_wc=output_shape)
    t_predict = time.time() - t_predict

    # write timing informations as textfile in the n5 topdir
    with open(os.path.join(out_file, 't-inf_gpu%i.txt' % gpu), 'w') as f:
        f.write("Inference with gpu %i in %f s" % (gpu, t_predict))


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    iteration = int(sys.argv[3])
    single_gpu_inference(sample, gpu, iteration)
