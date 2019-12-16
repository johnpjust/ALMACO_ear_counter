from tensorflow.keras import layers, Input, Model
import tensorflow as tf


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

with tf.device('/gpu:0'):
    encoder_input = Input(shape=(28, 28, 1), name='img')
    x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    encoder = Model(encoder_input, encoder_output, name='encoder')


    x = layers.Reshape((4, 4, 1))(encoder_output)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

    autoencoder = Model(encoder_input, decoder_output, name='autoencoder')

encoder.summary()
autoencoder.summary()

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
    x = layers.Dense(4, activation=actfun)(x)