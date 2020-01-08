import numpy as np
from scipy import ndimage, stats
from skimage import io
import glob
import os
import datetime
import pathlib
import json
import tensorflow as tf
from lr_scheduler_pt import *
from windowed import windowed
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

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
        self.conv_out = nn.Conv2d(kernel_size=3, in_channels=channel_num, out_channels=feat_out)
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
    def __init__(self, frames, cnn_feat_out=6):
        super(Combine, self).__init__()
        self.cnn = CNN(feat_out=cnn_feat_out)
        self.fc1 = nn.Linear(cnn_feat_out*frames, 10)
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


def train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args, empty_logs_train, empty_logs_val, empty_logs_test, criterion):
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        # print("train")
        train_loss = 0
        # for ind in range(len(data_loader_train)):
        for ind in np.random.permutation(len(data_loader_train)):
            # x_mb = tf.signal.frame(data_loader_train[ind][0], args.num_frames, 1, axis=0)
            # x_mb = np.array(list(more_itertools.windowed(data_loader_train[ind][0], n=args.num_frames, step=1)))
            # x_mb = windowed(data_loader_train[ind][0], n=args.num_frames, step=1)
            # y_mb = data_loader_train[ind][1]
            for i_ in range(2):
                if i_ == 0:
                    x_mb = windowed(data_loader_train[ind][0], n=args.num_frames, step=1)
                    y_mb = data_loader_train[ind][1]
                elif i_ == 1:
                    if np.random.choice(a=[False, True]):
                        x_mb = windowed(data_loader_train[ind][0][empty_logs_train[ind] == 0], n=args.num_frames, step=1)
                        y_mb = 0
                    else:
                        x_mb = np.repeat(data_loader_train[0][0], args.num_frames, axis=-1)
                        y_mb = 0
                    # x_mb = windowed(data_loader_train[ind][0][empty_logs_train[ind] == 0], n=args.num_frames, step=1)
                    # y_mb = 0
                # else:
                #     x_mb = np.repeat(data_loader_train[0][0], args.num_frames, axis=-1)
                #     y_mb = 0

                count = 0
                model.zero_grad()
                torch.cuda.empty_cache()
                for x in batch(x_mb, args.device, args.batch_size):
                    x = x.permute(0, 1, -1, -3, -2)
                    count += model(x).sum()
                # loss = criterion(count, y_mb)/y_mb ## make loss a percent
                loss = criterion(count, y_mb)
                loss.backward()
                # for p in model.parameters():
                #     p.grad /= x_mb.shape[0]  # scale accumulated gradients so each log has equal influence regardless of log length
                optimizer.step()

                if i_ == 0:
                    tf.summary.scalar('loss/train', loss**2, args.global_step)

        ## potentially update batch norm variables manuallu
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        # train_loss = tf.reduce_mean(train_loss)
        validation_loss = []
        validation_loss_static = []
        validation_loss_empty = []
        for ind in range(len(data_loader_val)):
            for i_ in range(3):
                if i_ == 0:
                    x_mb = windowed(data_loader_val[ind][0], n=args.num_frames, step=1)
                    y_mb = data_loader_val[ind][1]
                elif i_ == 1:
                    x_mb = windowed(data_loader_val[ind][0][empty_logs_val[ind] == 0], n=args.num_frames, step=1)
                else:
                    x_mb = np.repeat(data_loader_val[0][0], args.num_frames, axis=-1)
                count = 0
                model.zero_grad()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for x in batch(x_mb, args.device, args.batch_size):
                        x = x.permute(0, 1, -1, -3, -2)
                        count += model(x).sum()

                    if i_ == 0:
                        validation_loss.append(((count.cpu().numpy() - y_mb)**2))
                    elif i_ == 1:
                        validation_loss_empty.append(count.cpu().numpy()**2)
                    else:
                        validation_loss_static.append(count.cpu().numpy()**2)

        validation_loss = np.mean(validation_loss)
        validation_loss_empty = np.mean(validation_loss_empty)
        validation_loss_static = np.mean(validation_loss_static)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        test_loss_static = []
        test_loss_empty = []
        for ind in range(len(data_loader_test)):
            for i_ in range(3):
                if i_ == 0:
                    x_mb = windowed(data_loader_test[ind][0], n=args.num_frames, step=1)
                    y_mb = data_loader_test[ind][1]
                elif i_ == 1:
                    x_mb = windowed(data_loader_test[ind][0][empty_logs_test[ind] == 0], n=args.num_frames, step=1)
                else:
                    x_mb = windowed(data_loader_test[ind][0][empty_logs_test[ind] == 0], n=args.num_frames, step=1)

                count = 0
                for x_ in batch(x_mb, args.batch_size):
                    count += tf.reduce_sum(model(x_, training=False)).numpy()

                if i_ == 0:
                    test_loss.append(tf.math.squared_difference(count, y_mb))
                elif i_ == 1:
                    test_loss_empty.append(tf.math.square(count))
                else:
                    test_loss_static.append(tf.math.square(count))

        test_loss = tf.reduce_mean(test_loss)
        test_loss_empty = tf.reduce_mean(test_loss_empty)
        test_loss_static = tf.reduce_mean(test_loss_static)
        # print("test loss:  " + str(test_loss))

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)

        #### tensorboard
        # tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/validation', validation_loss, args.global_step)
        tf.summary.scalar('loss/validation_empty', validation_loss_empty, args.global_step)
        tf.summary.scalar('loss/validation_static', validation_loss_static, args.global_step)
        tf.summary.scalar('loss/test', test_loss, args.global_step) ##tf.compat.v1.train.get_global_step()
        tf.summary.scalar('loss/test_empty', test_loss_empty, args.global_step)  ##tf.compat.v1.train.get_global_step()
        tf.summary.scalar('loss/test_static', test_loss_static, args.global_step)  ##tf.compat.v1.train.get_global_step()

        if stop:
            break

def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))

class parser_:
    pass

args = parser_()
args.device = torch.device("cuda:0") #'/gpu:0'  # '/gpu:0'
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
args.global_step = 0

args.path = os.path.join(args.tensorboard,
                         'frames{}_{}'.format(args.num_frames,
                             str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

if not args.load:
    print('Creating directory experiment..')
    pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)


torch.cuda.set_device(args.device)

### import and setup data
data = np.array(np.load(r'D:\AlmacoEarCounts\almaco_earcount_labeled_data.npy', allow_pickle=True))
empty_logs=np.load(r'D:\AlmacoEarCounts\empty_log_indx.npy', allow_pickle=True)
bg=np.load(r'D:\AlmacoEarCounts\bg.npy')
data = np.array(data)
for imgs in data:
    imgs[0] = imgs[0] - bg

data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
empty_logs_train = empty_logs[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
data_loader_val = data[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
empty_logs_val = empty_logs[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
data_loader_test = data[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]
empty_logs_test = empty_logs[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]


# model.to(device)
# mytensor = my_tensor.to(device)

########### create model
model = Combine(frames=args.num_frames)
## check on GPU
# next(model.parameters()).is_cuda



###################################
## tensorboard and saving
writer = tf.summary.create_file_writer(args.path)
writer.set_as_default()
tf.compat.v1.train.get_or_create_global_step()

# global_step = tf.compat.v1.train.get_global_step()
# global_step.assign(0)

root = None
args.start_epoch = 0

print('Creating optimizer..')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


# root = tf.train.Checkpoint(optimizer=optimizer,
#                            model=model,
#                            optimizer_step=tf.compat.v1.train.get_global_step())
#
# if args.load:
#     load_model(args, root, load_start_epoch=True)
#
# print('Creating scheduler..')
# # use baseline to avoid saving early on
scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

with open(os.path.join(args.path, 'modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        summary(model.to(args.device), (args.num_frames, 3, 108, 192))

print("training")
train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args, empty_logs_train,
      empty_logs_val, empty_logs_test)

#### C:\Program Files\NVIDIA Corporation\NVSMI
#### tensorboard --logdir=D:\AlmacoEarCounts\Tensorboard

######### nvidia-smi  -l 2
