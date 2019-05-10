import os
import sys
import math
import importlib
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from pathlib import Path
from datetime import datetime

###############################################################################
# ARGUMENTS
###############################################################################

# Append python path
MODELS_PATH = 'logs_modelnet'
PATH_TRAIN = '../data/modelnet40_ply_hdf5_2048/train_files.txt'
PATH_VALID = '../data/modelnet40_ply_hdf5_2048/test_files.txt'
MODEL_NAME = 'pointcnn_cls'
SETTINGS_FILE = 'modelnet_x3_l4'

NUM_CLASSES = 40
BATCH_SIZE = 4

# Import model
model = importlib.import_module(MODEL_NAME)

# Import settings
setting_path = os.path.join(str(Path().resolve()), MODEL_NAME)
sys.path.append(setting_path)
setting = importlib.import_module(SETTINGS_FILE)

# List all settings
num_epochs = setting.num_epochs
batch_size = setting.batch_size if BATCH_SIZE is None else BATCH_SIZE
sample_num = setting.sample_num
step_val = setting.step_val
rotation_range = setting.rotation_range
rotation_range_val = setting.rotation_range_val
scaling_range = setting.scaling_range
scaling_range_val = setting.scaling_range_val
jitter = setting.jitter
jitter_val = setting.jitter_val

###############################################################################
# LOAD DATASET
###############################################################################

# Prepare inputs
print('{}-Preparing datasets...'.format(datetime.now()))
_, _, data_val, label_val = setting.load_fn(PATH_TRAIN, PATH_VALID)

# Info
num_train = data_val.shape[0]
point_num = data_val.shape[1]
num_val = data_val.shape[0]
print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))

###############################################################################
# PLACEHOLDERS
###############################################################################

# Placeholders
indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
jitter_range = tf.placeholder(tf.float32, shape=1, name="jitter_range")
global_step = tf.Variable(0, trainable=False, name='global_step')
is_training = tf.placeholder(tf.bool, name='is_training')

# Data placeholders
data_val_placeholder = tf.placeholder(data_val.dtype, data_val.shape, name='data_val')
label_val_placeholder = tf.placeholder(tf.int64, label_val.shape, name='label_val')
handle = tf.placeholder(tf.string, shape=[], name='handle')

# Iterator
iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.int64))
(pts_fts, labels) = iterator.get_next()

# Dataset
dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
if setting.map_fn is not None:
    dataset_val = dataset_val.map(lambda data, label: tuple(tf.py_func(
        setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
dataset_val = dataset_val.batch(batch_size)
batch_num_val = math.ceil(num_val / batch_size)
iterator_val = dataset_val.make_initializable_iterator()
print('{}-{:d} testing batches per test.'.format(datetime.now(), int(batch_num_val)))

# Points/features
pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
features_augmented = None
if setting.data_dim > 3:
    points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                [3, setting.data_dim - 3],
                                                axis=-1,
                                                name='split_points_features')
    if setting.use_extra_features:
        if setting.with_normal_feature:
            if setting.data_dim < 6:
                print('Only 3D normals are supported!')
                exit()
            elif setting.data_dim == 6:
                features_augmented = pf.augment(features_sampled, rotations)
            else:
                normals, rest = tf.split(features_sampled, [3, setting.data_dim - 6])
                normals_augmented = pf.augment(normals, rotations)
                features_augmented = tf.concat([normals_augmented, rest], axis=-1)
        else:
            features_augmented = features_sampled
else:
    points_sampled = pts_fts_sampled
points_augmented = pf.augment(points_sampled, xforms, jitter_range)

###############################################################################
# MODEL DEFINITION
###############################################################################

net = model.Net(points=points_augmented, features=features_augmented, is_training=is_training, setting=setting)
logits = net.logits
probs = tf.nn.softmax(logits, name='probs')
predictions = tf.argmax(probs, axis=-1, name='predictions')

labels_2d = tf.expand_dims(labels, axis=-1, name='labels_2d')
labels_tile = tf.tile(labels_2d, (1, tf.shape(logits)[1]), name='labels_tile')

###############################################################################
# EVAL METHOD
###############################################################################


def evaluate(model_num, num_votes, verbose=False):

    # Model path
    model_path = MODELS_PATH + '/model_' + str(model_num) + '/model_' + str(model_num)

    # Start session
    with tf.Session() as sess:

        # Load the model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        if verbose:
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), model_path))

        # Handle
        handle_val = sess.run(iterator_val.string_handle())

        # Data to remember
        voted_logits = []
        voted_labels = None

        # Num votes range
        for _ in range(num_votes):

            # Restart dataset iterator
            sess.run(iterator_val.initializer, feed_dict={
                        data_val_placeholder: data_val,
                        label_val_placeholder: label_val,
                    })

            print (data_val[0][0])
            exit()

            # Data to remember
            batch_logits = []
            batch_labels = []

            # For each batch in dataset
            for batch_idx_val in range(int(batch_num_val)):

                # Set the batchsize
                if not setting.keep_remainder or num_val % batch_size == 0 or batch_idx_val != batch_num_val - 1:
                    batch_size_val = batch_size
                else:
                    batch_size_val = num_val % batch_size

                # Get the xforms and rotations
                xforms_np, rotations_np = pf.get_xforms(batch_size_val, rotation_range=rotation_range_val,
                                                        scaling_range=scaling_range_val, order=setting.rotation_order)

                # Get logits and labels
                pred_val, labels_val = sess.run([logits, labels_tile], feed_dict={
                             handle: handle_val,
                             indices: pf.get_indices(batch_size_val, sample_num, point_num),
                             xforms: xforms_np,
                             rotations: rotations_np,
                             jitter_range: np.array([jitter_val]),
                             is_training: False,
                         })

                # Remember data
                batch_logits.append(np.squeeze(pred_val, axis=1))
                batch_labels.append(np.squeeze(labels_val))

            # Concatenate
            batch_logits = np.concatenate(batch_logits, axis=0)
            batch_labels = np.concatenate(batch_labels)
            voted_logits.append(batch_logits)
            voted_labels = batch_labels

        # Final concat
        voted_logits = np.sum(np.stack(voted_logits).transpose(1, 2, 0), axis=-1)
        voted_predic = np.argmax(voted_logits, axis=-1)
        voted_accura = float(np.sum(voted_predic == voted_labels)) / len(voted_labels)
        return voted_logits, voted_labels, voted_accura

###############################################################################
# TIME EVAL
###############################################################################


logits, labels, accuracy = evaluate(1, 1, True)
print('LOGITS:', logits.shape)
print('LABELS:', labels.shape)
print('ACCURACY:', accuracy)
