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
from windowed import windowed
from contextlib import redirect_stdout

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args, empty_logs_train, empty_logs_val, empty_logs_test):
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        # print("train")
        train_loss = 0
        # for ind in range(len(data_loader_train)):
        for ind in np.random.permutation(len(data_loader_train)):
            # x_mb = tf.signal.frame(data_loader_train[ind][0], args.num_frames, 1, axis=0)
            # x_mb = np.array(list(more_itertools.windowed(data_loader_train[ind][0], n=args.num_frames, step=1)))
            for i_ in range(3):
                if i_ == 1:
                    x_mb = windowed(data_loader_train[ind][0], n=args.num_frames, step=1)
                    y_mb = data_loader_train[ind][1]
                elif i_ == 0:
                    x_mb = windowed(data_loader_train[ind][0][empty_logs_train[ind] == 0], n=args.num_frames, step=1)
                    y_mb = 0
                else:
                    x_mb = np.repeat(data_loader_train[0][0], args.num_frames, axis=-1)
                    y_mb = 0

                count = 0
                batch = 0
                grads = [np.zeros_like(x) for x in model.trainable_variables]
                for x_ in batch(x_mb, args.batch_size):
                    with tf.GradientTape() as tape:
                        count_ = tf.reduce_sum(model(x_, training=True))
                    count += count_
                    grads_ = tape.gradient(count_, model.trainable_variables)
                    grads = [x1 + x2 for x1, x2 in zip(grads, grads_)]
                    batch += 1
                grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
                loss = count-y_mb
                ## scale gradients by log length or number of batches (else longer logs will be weighted unduly)
                globalstep = optimizer.apply_gradients(zip([2*loss*x/batch for x in grads], model.trainable_variables))

                tf.summary.scalar('loss/train', loss**2, globalstep)

        ## potentially update batch norm variables manuallu
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        # train_loss = tf.reduce_mean(train_loss)
        validation_loss=[]
        for ind in range(len(data_loader_val)):
            for i_ in range(2):
                if i_ == 1:
                    x_mb = windowed(data_loader_val[ind][0], n=args.num_frames, step=1)
                    y_mb = data_loader_val[ind][1]
                else:
                    x_mb = windowed(data_loader_val[ind][0][empty_logs_val[ind] == 0], n=args.num_frames, step=1)
                    y_mb = 0

                count = 0
                for x_ in batch(x_mb, args.batch_size):
                    count += tf.reduce_sum(model(x_, training=False)).numpy()
                validation_loss.append(tf.math.squared_difference(count, y_mb))
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        for ind in range(len(data_loader_test)):
            for i_ in range(2):
                if i_ == 1:
                    x_mb = windowed(data_loader_test[ind][0], n=args.num_frames, step=1)
                    y_mb = data_loader_test[ind][1]
                else:
                    x_mb = windowed(data_loader_test[ind][0][empty_logs_test[ind] == 0], n=args.num_frames, step=1)
                    y_mb = 0

                count = 0
                for x_ in batch(x_mb, args.batch_size):
                    count += tf.reduce_sum(model(x_, training=False)).numpy()
                test_loss.append(tf.math.squared_difference(count, y_mb))
        test_loss = tf.reduce_mean(test_loss)
        # print("test loss:  " + str(test_loss))

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)

        #### tensorboard
        # tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/validation', validation_loss, globalstep)
        tf.summary.scalar('loss/test', test_loss, globalstep) ##tf.compat.v1.train.get_global_step()

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
args.batch_size = 200 ## 6/50,
args.epochs = 5000
args.patience = 20
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
empty_logs=np.load(r'D:\AlmacoEarCounts\empty_log_indx.npy', allow_pickle=True)
data_ = np.array(data)
data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
empty_logs_train = empty_logs[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
data_loader_val = data[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
empty_logs_val = empty_logs[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
data_loader_test = data[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]
empty_logs_test = empty_logs[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]

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

bg=np.load(r'D:\AlmacoEarCounts\bg.npy')
for imgs in data_loader_train:
    imgs[0] = imgs[0] - bg
for imgs in data_loader_val:
    imgs[0] = imgs[0] - bg
for imgs in data_loader_test:
    imgs[0] = imgs[0] - bg


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
# with tf.device('/gpu:0'):
#     inputs = tf.keras.Input(shape=(108, 192, 3), name='img') ## (108, 192, 3)
#     x = layers.Conv2D(16, 3, activation='relu')(inputs)
#     x = layers.Conv2D(16, 3, activation='relu')(x)
#     block_1_output = layers.MaxPooling2D(2)(x)
#
#     x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_1_output)
#     # x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
#     x = layers.add([x, block_1_output])
#     block_2_output = layers.MaxPooling2D(2)(x)
#
#     x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_2_output)
#     x = layers.add([x, block_2_output])
#     block_3_output = layers.GlobalAveragePooling2D()(x)
#
#     cnn = tf.keras.Model(inputs, block_3_output, name='toy_resnet')
#
#     input_sequences = tf.keras.Input(shape=(6, 108, 192, 3)) ## (108, 192, 3)
#     x = layers.TimeDistributed(cnn)(input_sequences)
#     x = layers.Flatten()(x)
#     x = layers.Dense(16, activation='relu')(x)
#     x = layers.Dense(1)(x)
#     counts = tf.keras.activations.softplus(x)
#     model = tf.keras.Model(input_sequences, counts, name='toy_resnet')
#     model.summary()

# actfun = tf.nn.relu
# with tf.device('/gpu:0'):
#     inputs = tf.keras.Input(shape=(108, 192, 3*args.num_frames), name='img') ## (108, 192, 3)
#     x = layers.Conv2D(32, 3, activation=actfun)(inputs)
#     x = layers.Conv2D(32, 3, activation=actfun)(x)
#     block_1_output = layers.MaxPooling2D(2)(x)
#
#     x = layers.Conv2D(32, 3, activation=actfun, padding='same')(block_1_output)
#     # x = layers.Conv2D(32, 3, activation=actfun, padding='same')(x)
#     x = layers.add([x, block_1_output])
#     block_2_output = layers.MaxPooling2D(2)(x)
#
#     x = layers.Conv2D(32, 3, activation=actfun, padding='same')(block_2_output)
#     # x = layers.Conv2D(32, 3, activation=actfun, padding='same')(x)
#     x = layers.add([x, block_2_output])
#     # block_3_output = layers.MaxPooling2D(2)(x)
#     block_4_output = layers.GlobalAveragePooling2D()(x)
#     #
#     # x = layers.Conv2D(32, 3, activation=actfun, padding='same')(block_3_output)
#     # # x = layers.Conv2D(32, 3, activation=actfun, padding='same')(x)
#     # x = layers.add([x, block_3_output])
#     # block_4_output = layers.GlobalAveragePooling2D()(x)
#
#     x = layers.Flatten()(block_4_output)
#     x = layers.Dense(32, activation=actfun)(x)
#     x = layers.Dense(1)(x)
#     counts = tf.keras.activations.softplus(x)
#     model = tf.keras.Model(inputs, counts, name='toy_resnet')
#     model.summary()

actfun = tf.nn.relu
with tf.device('/gpu:0'):
    inputs = tf.keras.Input(shape=(108, 192, 3*args.num_frames), name='img') ## (108, 192, 3)
    x = layers.Conv2D(32, 7, activation=actfun, strides=2)(inputs)
    block_output = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(32, 1, activation=actfun)(x)
    block_output = layers.AveragePooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(32, 1, activation=actfun)(x)
    x = layers.AveragePooling2D(2, strides=2)

    x = layers.Conv2D(32, 1, activation=actfun)(block_output)
    x = layers.Conv2D(32, 3, activation=None)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation=actfun)(x)
    x = layers.Dense(1)(x)
    counts = tf.keras.activations.softplus(x)
    model = tf.keras.Model(inputs, counts, name='toy_resnet')
    model.summary()

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
scheduler = EarlyStopping(model=model, patience=args.patience, args=args, root=root)

with open(os.path.join(args.path, 'modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        model.summary()

with tf.device(args.device):
    train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args, empty_logs_train, empty_logs_val, empty_logs_test)

#### tensorboard --logdir=D:\AlmacoEarCounts\Tensorboard

########### C:\Program Files\NVIDIA Corporation\NVSMI
######### nvidia-smi  -l 2


dataset = data_loader_test
validation_loss = []
for ind in range(len(dataset)):
    x_mb = windowed(dataset[ind][0], n=args.num_frames, step=1)
    y_mb = dataset[ind][1]
    count = []
    for x_ in batch(x_mb, args.batch_size):
        count.extend(model(x_, training=False).numpy())
    validation_loss.append(count.copy())

validation_loss = [np.squeeze(np.array(x__)) for x__ in validation_loss]
with open(os.path.join(args.path, 'test.csv'),'w') as file:
    for line in validation_loss:
        file.write(','.join([str(x__) for x__ in line]))
        file.write('\n')

with open(os.path.join(args.path, 'test_y.csv'),'w') as file:
    for line in dataset:
        file.write(','.join([str(x__) for x__ in line[1:]]))
        file.write('\n')


## find empty conveyor sequences
dist = [np.sqrt(np.sum(np.square(x__))) for x__ in data_loader_val[0][0]]