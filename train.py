import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

num_frames = 6
### import and setup data
### tf.signal.frame
data = np.load(r'D:\AlmacoEarCounts\almaco_earcount_labeled_data.npy', allow_pickle=True)
data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:100]]
data_loader_val = data[data_split_ind[100:]]

##### use keras functional api  https://www.tensorflow.org/guide/keras/functional
##### toy autoencoder & resnet model examples, among other things

######### Create Model
###### CNN feature extractor
inputs = tf.keras.Input(shape=data_loader_train[0][0][0].shape, name='img') ## (108, 192, 3)
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(32, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])
block_2_output = layers.MaxPooling2D(2)(block_2_output)

x = layers.Conv2D(32, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])
block_3_output = layers.MaxPooling2D(2)(block_3_output)

x = layers.Conv2D(32, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)
x = tf.signal.frame(x,num_frames,1, axis=0)
x = layers.Flatten()(x)
x = layers.Dense(20, activation='relu')(x)
x = layers.Dense(1)(x)
counts = tf.keras.activations.softplus(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, counts, name='toy_resnet')
model.summary()

###### dense layer state tracking/transitions

#########################################
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
#
# inputs = tf.keras.Input(shape=(3,))
# x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
# outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)