import numpy as np
import tensorflow as tf
from scipy import ndimage, stats
from skimage import io
import glob
import os
import datetime
import pathlib
import json
from lr_scheduler import *
import more_itertools
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, channel_num=32, feat_out=6):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channel_num, kernel_size=7, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=channel_num,out_channels=channel_num, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(in_channels=channel_num, out_channels=feat_out)
        self.fc1 = nn.Linear(channel_num, channel_num//2)
        self.fc_out = nn.Linear(channel_num//2, feat_out)

    def forward(self, x): ##input (108, 192, 3)
        block_output = self.maxpool1(F.relu(self.conv1(x)))

        x = F.relu(self.conv2(block_output))
        x = block_output + self.conv3(x)
        x = F.relu(self.conv2(x))
        block_output = self.avgpool1(x)

        x = F.relu(self.conv2(block_output))
        x = block_output + self.conv3(x)
        x = F.relu(self.conv2(x))
        block_output = self.avgpool1(x)

        x = F.relu(self.conv2(block_output))
        x = self.conv4(x)

        ###### dense output
        x = F.adaptive_avg_pool2d(x, (1, 1)) ## global average pooling
        # x = F.adaptive_max_pool2d(x, (1,1))
        x = x.view(x.size()[0], -1) ## flatten
        x = F.elu(self.fc1(x))
        out = self.fc_out(x)

        # ###### conv output
        # x = self.conv_out(x)
        # out = F.adaptive_avg_pool2d(x, (1, 1)) ## global average pooling

        return out

## net = Net()
class Combine(nn.Module):
    def __init__(self, cnn_feat_out=6):
        super(Combine, self).__init__()
        self.cnn = CNN()
        self.fc1 = nn.Linear(cnn_feat_out, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        ## reshape to simulate TF time-distributed-layer
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        # fc1_in = c_out.view(batch_size, timesteps, -1)
        fc1_in = c_out.view(batch_size, -1)
        fc1_out = F.elu(self.fc1(fc1_in))
        fc2_out = self.fc2(fc1_out)
        return F.softplus(fc2_out)

def batch(iterable, device, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield torch.from_numpy(iterable[ndx:min(ndx + n, l)]).float().to(device)

def train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args):
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        print("train")
        train_loss = 0
        # for ind in range(len(data_loader_train)):
        for ind in np.random.permutation(len(data_loader_train)):
            # x_mb = tf.signal.frame(data_loader_train[ind][0], args.num_frames, 1, axis=0)
            x_mb = np.array(list(more_itertools.windowed(data_loader_train[ind][0], n=args.num_frames, step=1)))
            y_mb = data_loader_train[ind][1]
            count = 0
            grads = [np.zeros_like(x) for x in model.trainable_variables]

            for x_ in batch(x_mb, args.device, args.batch_size):
                x_ = torch.from_numpy(x_).float().to(args.device)
                with tf.GradientTape() as tape:
                    count_ = tf.reduce_sum(model(x_))
                count += count_
                grads_ = tape.gradient(count_, model.trainable_variables)
                grads = [x1 + x2 for x1, x2 in zip(grads, grads_)]

            # grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            loss = count-y_mb
            globalstep = optimizer.apply_gradients(zip([2*loss*x for x in grads], model.trainable_variables))

            tf.summary.scalar('loss/train', loss**2, globalstep)

        ## potentially update batch norm variables manuallu
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        # # train_loss = tf.reduce_mean(train_loss)
        # print("val")
        # validation_loss=[]
        # for ind in range(len(data_loader_val)):
        #     x_mb = np.array(list(more_itertools.windowed(data_loader_val[ind][0], n=args.num_frames, step=1)))
        #     y_mb = data_loader_val[ind][1]
        #     count = 0
        #     for x_ in batch(x_mb, args.batch_size):
        #         count += tf.reduce_sum(model(x_)).numpy()
        # #     validation_loss.append(tf.math.squared_difference(count, y_mb))
        # # validation_loss = tf.reduce_mean(validation_loss)
        # # print("validation loss:  " + str(validation_loss))
        #
        # print("test")
        # test_loss=[]
        # for ind in range(len(data_loader_test)):
        #     x_mb = np.array(list(more_itertools.windowed(data_loader_test[ind][0], n=args.num_frames, step=1)))
        #     y_mb = data_loader_test[ind][1]
        #     count = 0
        #     for x_ in batch(x_mb, args.batch_size):
        #         count += tf.reduce_sum(model(x_)).numpy()
        # #     test_loss.append(tf.math.squared_difference(count, y_mb))
        # # test_loss = tf.reduce_mean(test_loss)
        # # print("test loss:  " + str(test_loss))
        # #
        # # stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)
        # #
        # # #### tensorboard
        # # # tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        # # tf.summary.scalar('loss/validation', validation_loss, tf.compat.v1.train.get_global_step())
        # # tf.summary.scalar('loss/test', test_loss, tf.compat.v1.train.get_global_step())
        # #
        # # if stop:
        # #     break

def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))

class parser_:
    pass

args = parser_()
args.device = 'cuda' #'/gpu:0'  # '/gpu:0'
args.learning_rate = np.float32(1e-2)
args.clip_norm = 0.1
args.batch_size = 20 ## 6/50,
args.epochs = 5000
args.patience = 10
args.load = ''
args.tensorboard = r'D:\AlmacoEarCounts\Tensorboard'
args.early_stopping = 500
args.manualSeed = None
args.manualSeedw = None
args.p_val = 0.2
args.num_frames = 6

# args.path = os.path.join(args.tensorboard,
#                          'frames{}_{}'.format(args.num_frames,
#                              str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))
#
# if not args.load:
#     print('Creating directory experiment..')
#     pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
#     with open(os.path.join(args.path, 'args.json'), 'w') as f:
#         json.dump(str(args.__dict__), f, indent=4, sort_keys=True)
#
# ### import and setup data
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
args.device = torch.device("cuda:0")

# model.to(device)
# mytensor = my_tensor.to(device)

########### create model
net = Net()
summary(net, (3, 108, 192))

crnn = TimeDistributed(net)
crnn.to(args.device)
summary(crnn, (3, 108, 192))


###################################
## tensorboard and saving
# writer = tf.summary.create_file_writer(args.path)
# writer.set_as_default()
# tf.compat.v1.train.get_or_create_global_step()
#
# global_step = tf.compat.v1.train.get_global_step()
# global_step.assign(0)
#
# root = None
# args.start_epoch = 0
#
print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()
# root = tf.train.Checkpoint(optimizer=optimizer,
#                            model=model,
#                            optimizer_step=tf.compat.v1.train.get_global_step())
#
# if args.load:
#     load_model(args, root, load_start_epoch=True)
#
# print('Creating scheduler..')
# # use baseline to avoid saving early on
# scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)
#
# with tf.device(args.device):
#     train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args)

#### C:\Program Files\NVIDIA Corporation\NVSMI
#### tensorboard --logdir=D:\AlmacoEarCounts\Tensorboard

######### nvidia-smi  -l 2
