import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy import ndimage, stats
from skimage import io
import glob

num_frames = 6
### import and setup data
data = np.load(r'D:\AlmacoEarCounts\almaco_earcount_labeled_data.npy', allow_pickle=True)
data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:100]]
data_loader_val = data[data_split_ind[100:]]

########## get image background
bg=np.load(r'D:\AlmacoEarCounts\bg.npy')
### WestBilslandP0007 50:62 --> img 56
for x in data_loader_val:
    if x[-1] == 'WestBilslandP0007':
        break

# bgs = []
# for x in data_loader_val:
#     imgs = x[0]
#     img_mode = stats.mode(imgs, axis=0)
#     bg = img_mode.mode.squeeze()
#     bg = ndimage.median_filter(bg, (5,5,3))
#     bgs.append(bg)
##### use keras functional api  https://www.tensorflow.org/guide/keras/functional
##### toy autoencoder & resnet model examples, among other things

for imgs in data_loader_train:
    imgs[0] = imgs[0] - bg
for imgs in data_loader_val:
    imgs[0] = imgs[0] - bg
######### Create Model
###### CNN feature extractor
inputs = tf.keras.Input(shape=data_loader_train[0][0][0].shape, name='img') ## (108, 192, 3)
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(16, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_1_output)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])
block_2_output = layers.MaxPooling2D(2)(block_2_output)

x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_2_output)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])
block_3_output = layers.MaxPooling2D(2)(block_3_output)

x = layers.Conv2D(32, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)
x = tf.signal.frame(x,num_frames,1, axis=0)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(1)(x)
counts = tf.keras.activations.softplus(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, counts, name='toy_resnet')
model.summary()

