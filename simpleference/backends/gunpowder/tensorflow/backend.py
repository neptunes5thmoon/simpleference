import numpy as np
import os
import threading
# we try to use the tensorflow from gunpowder,
# otherwise we try to revert to normal tensorflow
try:
    from gunpowder.ext import tensorflow as tf
except ImportError:
    import tensorflow as tf


class TensorflowPredict(object):
    '''Tensorflow implementation of :class:`gunpowder.nodes.Predict`.

    Args:

        meta_graph_basename: Basename of a tensorflow meta-graph storing the
            trained tensorflow graph (aka a checkpoint), as created by
            :class:`gunpowder.nodes.Train`, for example.

        input_key (string): Name of the input layer.

        outputs (string): Name of the output layer.
    '''

    # TODO add gpu number as argument
    def __init__(self,
                 weight_graph_basename,
                 inference_graph_basename,
                 input_keys,
                 output_keys):
        assert os.path.exists(weight_graph_basename + '.index'), weight_graph_basename
        # NOTE this seems a bit dubious, don't know if this is persistent
        # for different tf models
        # assert os.path.exists(weight_graph_basename + '.data-00000-of-00001')
        self.weight_graph_basename = weight_graph_basename

        assert os.path.exists(inference_graph_basename + '.meta'), inference_graph_basename
        self.inference_graph_basename = inference_graph_basename
        if not (isinstance(input_keys, tuple) or isinstance(input_keys, list)):
            input_keys = [input_keys, ]
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self._read_meta_graph()

        self.lock = threading.Lock()

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray) or isinstance(input_data, list) or isinstance(input_data, tuple)
        if isinstance(input_data, np.ndarray):
            input_data = [input_data,]
        # we need to lock the inference on the gpu to prevent dask from running multiple predictions in
        # parallel. It might be beneficial, to only lock the inference step, but not to lock
        # shipping data onto / from the gpu.
        # Unfortunately I don't now how to do this in tf.
        with self.lock:
            output = self.session.run(self.output_keys, feed_dict=dict(zip(self.input_keys, input_data)))

        if isinstance(self.output_keys, list) or isinstance(self.output_keys, tuple):
            output_32 = []
            for o in output:
                output_32.append(o.astype('float32'))
        else:
            output_32 = np.array(output).astype('float32')
        return output_32

    def _read_meta_graph(self):
        # read the meta-graph
        saver = tf.train.import_meta_graph(self.inference_graph_basename + '.meta',
                                           clear_devices=True)
        # restore variables
        saver.restore(self.session, self.weight_graph_basename)

    # Needs to be called in the end
    def stop(self):
        if self.session is not None:
            self.session.close()
            self.graph = None
            self.session = None
