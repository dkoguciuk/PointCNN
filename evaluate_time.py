import os
import sys
import time
import math
import socket
import shutil
import random
import argparse
import importlib
import data_utils
import scipy.misc
import statistics 
import numpy as np
import pointfly as pf
import tensorflow as tf
from scipy import stats
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


###############################################################################
# ARGUMENTS
###############################################################################


# Append python path
MODELS_PATH = '../models/cls'
PATH_TRAIN = '../data/modelnet/train_files.txt'
PATH_VALID = '../data/modelnet/test_files.txt'
MODEL_NAME = 'pointcnn_cls'
SETTINGS_FILE = 'modelnet_x3_l4'
NUM_CLASSES = 40
BATCH_SIZE = 1

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
pool_setting_val = None if not hasattr(setting, 'pool_setting_val') else setting.pool_setting_val
pool_setting_train = None if not hasattr(setting, 'pool_setting_train') else setting.pool_setting_train

###############################################################################
# LOAD DATASET
###############################################################################

# Prepare inputs
print('{}-Preparing datasets...'.format(datetime.now()))
_, _, data_val, label_val = setting.load_fn(PATH_TRAIN, PATH_VALID)

# Info
num_val = data_val.shape[0]
point_num = data_val.shape[1]
print('{}-{:d} validation samples.'.format(datetime.now(), num_val))

###############################################################################
# PLACEHOLDERS
###############################################################################

# Placeholders
indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
global_step = tf.Variable(0, trainable=False, name='global_step')
is_training = tf.placeholder(tf.bool, name='is_training')

data_val_placeholder = tf.placeholder(data_val.dtype, data_val.shape, name='data_val')
label_val_placeholder = tf.placeholder(tf.int64, label_val.shape, name='label_val')
handle = tf.placeholder(tf.string, shape=[], name='handle')

dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
if setting.map_fn is not None:
    dataset_val = dataset_val.map(lambda data, label: tuple(tf.py_func(
        setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
if setting.keep_remainder:
    dataset_val = dataset_val.batch(batch_size)
    batch_num_val = math.ceil(num_val / batch_size)
else:
    dataset_val = dataset_val.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    batch_num_val = math.floor(num_val / batch_size)
iterator_val = dataset_val.make_initializable_iterator()
print('{}-{:d} testing batches per test.'.format(datetime.now(), int(batch_num_val)))

iterator = tf.data.Iterator.from_string_handle(handle, dataset_val.output_types)
(pts_fts, labels) = iterator.get_next()

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
loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=logits)

#with tf.name_scope('metrics'):
#    loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
#    print (loss_mean_op, loss_mean_update_op)
#    t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(labels_tile, predictions)
#    print (t_1_acc_op, t_1_acc_update_op)
#    t_1_per_class_acc_op, t_1_per_class_acc_update_op = tf.metrics.mean_per_class_accuracy(labels_tile,
#                                                                                           predictions,
#                                                                                           setting.num_class)
#    print (t_1_per_class_acc_op, t_1_per_class_acc_update_op)

#reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
#                                             if var.name.split('/')[0] == 'metrics'])

#_ = tf.summary.scalar('loss/train', tensor=loss_mean_op, collections=['train'])
#_ = tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])
#_ = tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

#_ = tf.summary.scalar('loss/val', tensor=loss_mean_op, collections=['val'])
#_ = tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_op, collections=['val'])
#_ = tf.summary.scalar('t_1_per_class_acc/val', tensor=t_1_per_class_acc_op, collections=['val'])

#lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
#                                       setting.decay_rate, staircase=True)
#lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
#_ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])
#reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
#if setting.optimizer == 'adam':
#    optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
#elif setting.optimizer == 'momentum':
#    optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
#    train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

#saver = tf.train.Saver(max_to_keep=None)
parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

###############################################################################
# EVAL METHOD
###############################################################################

def evaluate(model_num, num_votes, verbose=False):

    # Model path
    model_path = MODELS_PATH + '/model_' + str(model_num) + '/model_' + str(model_num)

    # Start session
    with tf.Session() as sess:
        #sess.run(init_op)

        # Load the model
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)
        if verbose:
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), model_path))
        print ("LOADED!")
        exit()

        # Handle
        handle_val = sess.run(iterator_val.string_handle())

        # Reset metrics
        #sess.run(reset_metrics_op)

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

                print ("PREDICTED", pred_val)
                print ("VALUE", labels_val)
                exit()

                # Remember data
                batch_logits.append(np.squeeze(pred_val, axis=1))
                batch_labels.append(np.squeeze(labels_val))


            # Concatenate
            batch_logits = np.concatenate(batch_logits, axis=0)
            batch_labels = np.concatenate(batch_labels, axis=0)
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

evaluate(1, 1, True)
exit()

###############################################################################
# NUM_VOTES DEPENDENCY
###############################################################################

#NUM_VOTES = 15
#RANGE_MODELS = range(1, 11)
#num_votes_accs = {i: [] for i in RANGE_MODELS}
#for i in num_votes_accs:
#    for x in range(1, NUM_VOTES+1):
#        _, _, acc = evaluate(model_num=i, num_votes=x, verbose=False)
#        num_votes_accs[i].append(acc)
#        print ('i=', i, 'x=', x, 'acc = ', acc)

#for i in num_votes_accs:
#    plt.plot(np.arange(1, 1+len(num_votes_accs[i])), num_votes_accs[i])
#    plt.savefig('num_votes_accs.png')

#num_votes_accs_np = np.zeros((max(RANGE_MODELS), NUM_VOTES), dtype=np.float)
#for i in num_votes_accs:
#    num_votes_accs_np[i-1] = num_votes_accs[i]
#    np.save('num_votes_accs.npy', num_votes_accs_np)

###############################################################################
# Calculate probabilities for each test cloud and each model.
# The output probability array will be the shape of (N, 40, X),
# where N is the test cloud len and X is the models count to be ensembled
###############################################################################

#NUM_MODELS = 10
#NUM_VOTES = 12
#probabilities = []
#true_labels = []
#accuracies = []
#for x in range(1, NUM_MODELS+1):
#    pred_vals, true_vals, acc = evaluate(model_num=x, num_votes=NUM_VOTES, verbose=False)
#    probabilities.append(pred_vals)
#    accuracies.append(acc)
#    true_labels = np.array(true_vals)
#    print ('Model =', x, 'acc = ', acc)
    
#probabilities = np.stack(probabilities).transpose(1, 2, 0)
#accuracies = np.array(accuracies)

#np.save('probabilities.npy', probabilities)
#np.save('true_labels.npy', true_labels)
#np.save('accuracies.npy', accuracies)

###############################################################################
# MODELS EVALUATION STATISTICS
###############################################################################

probabilities = np.load('probabilities.npy')
true_labels = np.load('true_labels.npy')
accuracies = np.load('accuracies.npy')

print('Mean accuracy =', statistics.mean(accuracies))					# 0.9181523500810373
indices = {}
for k in range(40):
    indices[k] = [i for i, x in enumerate(true_labels) if x == k]
    
validation_max_res = []
test_res_at_validation_max = []

for _ in range(1000):
    validation_indices = []
    test_indices = []
    for k in indices:
        random.shuffle(indices[k])
        split_idx = int(len(indices[k])/2)
        validation_indices += indices[k][:split_idx]
        test_indices += indices[k][split_idx:]
    validation_indices = sorted(validation_indices)
    test_indices = sorted(test_indices)

    validation_true_labels = true_labels[validation_indices]
    validation_probabilities = probabilities[validation_indices]
    test_true_labels = true_labels[test_indices]
    test_probabilities = probabilities[test_indices]

    validation_predictions = np.argmax(validation_probabilities, axis=1)
    validation_compare = np.equal(validation_predictions, np.expand_dims(validation_true_labels, -1))
    validation_accuracies = np.mean(validation_compare, axis=0)

    test_predictions = np.argmax(test_probabilities, axis=1)
    test_compare = np.equal(test_predictions, np.expand_dims(test_true_labels, -1))
    test_accuracies = np.mean(test_compare, axis=0)

    validation_max_res.append(np.max(validation_accuracies))
    test_res_at_validation_max.append(test_accuracies[np.argmax(validation_accuracies)])
    
mean_valid_max = statistics.mean(validation_max_res)
mean_test_at_valid_max = statistics.mean(test_res_at_validation_max)

print('Mean of validation max results =', mean_valid_max)				# 0.9222366288492707
print('Mean of test result for validation max =', mean_test_at_valid_max)		# 0.916160453808752

#Agregate the outputs with sum operation
aggregated_probability = np.sum(probabilities, axis=-1)
aggregated_predictions = np.argmax(aggregated_probability, axis=-1)
print ("SUM:", float(np.sum(aggregated_predictions == true_labels)) / len(true_labels)) # 0.9222042139384117

# Agregate the outputs with mean operation
aggregated_probability = np.mean(probabilities, axis=-1)
aggregated_predictions = np.argmax(aggregated_probability, axis=-1)
print ("MEAN", float(np.sum(aggregated_predictions == true_labels)) / len(true_labels)) # 0.9222042139384117

# Agregate the outputs with mode operation
aggregated_predictions = np.argmax(probabilities, axis=1)
aggregated_predictions =  np.squeeze(stats.mode(aggregated_predictions, axis=1)[0])
print ("MODE", float(np.sum(aggregated_predictions == true_labels)) / len(true_labels)) # 0.9222042139384117


