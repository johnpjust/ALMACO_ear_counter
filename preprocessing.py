import glob
import tensorflow as tf
import pathlib
import os
import string
import numpy as np

def get_label(file_path):
    # convert the path to a list of path components
    foldname = file_path.split(os.path.sep)[-1]
    # The second to last is the class-directory
    parts = foldname.split("_")
    return int(parts[-1]), parts[-4].replace(' ', '') + parts[-2]

def decode_img(img):
    img_size = 1.0 ## % resizing
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    if img_size < 1:
        imresize_ = tf.cast(tf.multiply(tf.cast(img.shape[:2], tf.float32), tf.constant(img_size)), tf.int32)
        img = tf.image.resize(img, size=imresize_, preserve_aspect_ratio=True)
        return img
    else:
        return img

def process_path(folder_path):
    ## idx is unique identifier for log
    label, idx = get_label(folder_path)
    # load the raw data from the file as a string
    imgs=[]
    for file in glob.glob(os.path.join(folder_path, '*.jpg')):
        img = tf.io.read_file(file)
        img = decode_img(img)
        imgs.append(img)
    return np.stack(imgs), label, idx

data_dir = glob.glob(r'T:\current\Projects\Deere\Harvester\Internal\HD Yield\2019\2019 Plot Combine Data\IA\AlmacoEarCounting\*')
# allfiles = glob.glob(r'T:\current\Projects\Deere\Harvester\Internal\HD Yield\2019\2019 Plot Combine Data\IA\AlmacoEarCounting\**\*.jpg', recursive=True)
out = []
for dir in data_dir[47:]:
    out.append(process_path(dir))
    print("done   " + out[-1][-1])


# imgs, labels, ids = [process_path(x) for x in data_dir]

######### create a tf.data.Dataset.from_tensor_slices object based on just the number of logs.  Then use it as a selector to preprocess the logs which are broadcast to the preprocessing function in different threads.

######## try reducing dimensionality and/or subtracting pixel modes to ease training process ##################

##### consider dequantizing images during training #####

###### tf.signal.frame(np.random.uniform(size=(10,3)),3,1, axis=0) #############
import pickle
fp_ = r'C:\Users\justjo\Desktop\almaco_earcount_labeled_data.pkl'
with open(fp_, "wb") as fp:   #Pickling
    pickle.dump(out, fp)