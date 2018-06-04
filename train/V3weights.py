from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file
from modelV3 import Deeplabv3

def get_xception_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.
    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('xception_65_', '')
    filename = filename.replace('decoder_', '', 1)
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
        return None
    if 'entry_flow' in filename or 'exit_flow' in filename:
        filename = filename.replace('_unit_1_xception_module', '')
    elif 'middle_flow' in filename:
        filename = filename.replace('_block1', '')
        filename = filename.replace('_xception_module', '')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'
def extract_tensors_from_checkpoint_file(filename, output_folder='weights', net_name=None):
    """Extract tensors from a TF checkpoint file.
    # Arguments
        filename: TF checkpoint file
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # convert tensor name into the corresponding Keras layer weight name and save
        if net_name == 'xception':
            filename = get_xception_filename(key)
        if filename:
            path = os.path.join(output_folder, filename)
            arr = reader.get_tensor(key)
            np.save(path, arr)
            print("tensor_name: ", key)
if __name__=='__main__':
    # extract_tensors_from_checkpoint_file(
    #     'xception/model.ckpt', net_name='xception', output_folder='xception')
    model = Deeplabv3(input_shape=(512, 512, 3),
                      classes=15, backbone='xception', weights=None)

    WEIGHTS_DIR = 'xception'
    print('Loading weights from', WEIGHTS_DIR)
    k=0
    for layer in model.layers[:359]:
        if layer.weights:
            weights = []
            for w in layer.weights:
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
                weights.append(weight_arr)
            layer.set_weights(weights)
        print(k)
        k=k+1

    print('Saving model weights...')
    model.save_weights('v3model.h5')
