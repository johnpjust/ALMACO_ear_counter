import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy import ndimage, stats
from skimage import io
import glob
import os
import datetime
import pathlib
import json
from lr_scheduler import *
import more_itertools

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args):
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        train_loss = 0
        for ind in np.random.permutation(len(data_loader_train)):
            # x_mb = tf.signal.frame(data_loader_train[ind][0], args.num_frames, 1, axis=0)
            x_mb = np.array(list(more_itertools.windowed(data_loader_train[ind][0], n=args.num_frames, step=1)))
            y_mb = data_loader_train[ind][1]
            count = 0
            grads = [tf.zeros_like(x) for x in model.trainable_variables]
            # gen = batch(x_mb, args.batch_size)
            # x_ = next(gen)
            # print("x_mb size:  " + str(x_mb.shape))
            for x_ in batch(x_mb, args.batch_size):
                with tf.GradientTape() as tape:
                    count_ = tf.reduce_sum(model(x_))
                count += count_
                grads_ = tape.gradient(count_, model.trainable_variables)
                grads = [x1 + x2 for x1, x2 in zip(grads, grads_)]

            # grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            loss = count-y_mb
            train_loss += tf.math.square(loss)
            global_step = optimizer.apply_gradients(zip([2*loss*x for x in grads], model.trainable_variables))

            tf.summary.scalar('loss/train', loss**2, tf.compat.v1.train.get_global_step())

            tf.compat.v1.train.get_global_step().assign_add(1)

        ## potentially update batch norm variables manuallu
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        train_loss = tf.reduce_mean(train_loss)
        validation_loss=0
        for ind in range(len(data_loader_val)):
            x_mb = np.array(list(more_itertools.windowed(data_loader_val[ind][0], n=args.num_frames, step=1)))
            y_mb = data_loader_val[ind][1]
            count = 0
            for x_ in batch(x_mb, args.batch_size):
                count_
                validation_loss += tf.math.squared_difference(tf.reduce_sum(model(x_mb)), y_mb)

        test_loss=0
        for ind in range(len(data_loader_test)):
            x_mb = np.array(list(more_itertools.windowed(data_loader_test[ind][0], n=args.num_frames, step=1)))
            y_mb = data_loader_test[ind][1]
            test_loss += tf.math.squared_difference(tf.reduce_sum(model(x_mb)), y_mb)

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)

        #### tensorboard
        tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/validation', validation_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/test', test_loss, tf.compat.v1.train.get_global_step())

        if stop:
            break

def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))

class parser_:
    pass

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.learning_rate = np.float32(1e-2)
args.clip_norm = 0.1
args.batch_size = 10 ## 6/50,
args.epochs = 5000
args.patience = 10
args.load = ''
args.tensorboard = r'D:\AlmacoEarCounts\Tensorboard'
args.early_stopping = 500
args.manualSeed = None
args.manualSeedw = None
args.p_val = 0.2
args.num_frames = 6

args.path = os.path.join(args.tensorboard,
                         'frames{}_{}'.format(args.num_frames,
                             str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

if not args.load:
    print('Creating directory experiment..')
    pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

### import and setup data
data = np.array(np.load(r'D:\AlmacoEarCounts\almaco_earcount_labeled_data.npy', allow_pickle=True))
data_ = np.array(data)
data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
data_loader_val = data[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
data_loader_test = data[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]

########## get image background
# for x in data_loader_train:
#     if x[-1] == 'WestBilslandP0007':
#         break

# bgs = []
# for x in data_loader_val:
#     imgs = x[0]
#     img_mode = stats.mode(imgs, axis=0)
#     bg = img_mode.mode.squeeze()
#     bg = ndimage.median_filter(bg, (5,5,3))
#     bgs.append(bg)
##### use keras functional api  https://www.tensorflow.org/guide/keras/functional
##### toy autoencoder & resnet model examples, among other things

# bg=np.load(r'D:\AlmacoEarCounts\bg.npy')
# for imgs in data_loader_train:
#     imgs[0] = imgs[0] - bg
# for imgs in data_loader_val:
#     imgs[0] = imgs[0] - bg

########### setup GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

######### Create Model
with tf.device('/gpu:0'):
    inputs = tf.keras.Input(shape=(108, 192, 3), name='img') ## (108, 192, 3)
    x = layers.Conv2D(16, 3, activation='relu')(inputs)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_1_output)
    # x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.add([x, block_1_output])
    block_2_output = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_2_output)
    x = layers.add([x, block_2_output])
    block_3_output = layers.GlobalAveragePooling2D()(x)

    cnn = tf.keras.Model(inputs, block_3_output, name='toy_resnet')

    input_sequences = tf.keras.Input(shape=(6, 108, 192, 3)) ## (108, 192, 3)
    x = layers.TimeDistributed(cnn)(input_sequences)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)
    counts = tf.keras.activations.softplus(x)
    model = tf.keras.Model(input_sequences, counts, name='toy_resnet')
    model.summary()

# for x_data in batch(np.random.uniform(size=(100,108,192,3)).astype(np.float32), 10):
# for n in range(50):
#     ##### exchange more_itertools with tf.signal.frame to get memory leak
#     x_mb = tf.signal.frame(np.random.uniform(size=(200,108,192,3)).astype(np.float32), args.num_frames, 1, axis=0)
#     for x_ in batch(x_mb, 10):
#     ################# no memory leak with more_itertools for sliding window framing ###############
#     # for x_ in batch(np.array(list(more_itertools.windowed(np.random.uniform(size=(100, 108, 192, 3)).astype(np.float32), n=6, step=1))),10):

###################################
## tensorboard and saving
writer = tf.summary.create_file_writer(args.path)
writer.set_as_default()
tf.compat.v1.train.get_or_create_global_step()

global_step = tf.compat.v1.train.get_global_step()
global_step.assign(0)

root = None
args.start_epoch = 0

print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.compat.v1.train.get_global_step())

if args.load:
    load_model(args, root, load_start_epoch=True)

print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

with tf.device(args.device):
    train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args)

#### tensorboard --logdir=D:\AlmacoEarCounts\Tensorboard

######### nvidia-smi  -l 2