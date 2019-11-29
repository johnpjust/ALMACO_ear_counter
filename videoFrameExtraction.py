import numpy as np
import cv2
import scipy.stats as st
import matplotlib.pyplot as plt
import glob
import pathlib
import pandas as pd
import os
import re

baseExtractpath = r'T:\current\Projects\Deere\Harvester\Internal\HD Yield\2019\2019 Plot Combine Data\IA\AlmacoEarCounting'

vidFiles = glob.glob(r'T:\current\Projects\Deere\Harvester\Internal\HD Yield\2019\2019 Plot Combine Data\IA\Yield Data\Ear Count Testing 2019-11-25\Videos_2019-11-25\*.avi')
# vidFiles = glob.glob(r'C:\Users\justjo\Desktop\New folder\*.avi')

gt = pd.read_excel(r'T:\current\Projects\Deere\Harvester\Internal\HD Yield\2019\2019 Plot Combine Data\IA\Yield Data\Ear Count Testing 2019-11-25\2019-11-20_Almaco_EarCounting_GTCounts.xlsx', sheet_name=1)
lognums = gt['Plot Maestro Log'].values
earcounts = gt['Total Ear Count'].values
lognumsfromFilenames = [int(re.search(r'(?<=P)\d+', x)[0]) for x in vidFiles]
loginxmatch = [lognumsfromFilenames.index(i) for i in lognums] ## should throw an error if can't find a file associated with ground truth
vidFiles = [vidFiles[i] for i in loginxmatch] ## now order of lognames matches ground truth file
count = 0
### redo log 76 (count=30)
for file in vidFiles:
    cap = cv2.VideoCapture(file)
    success,image = cap.read()
    imgs = []
    scale_percent = 10  # percent of original size 10%=(108, 192, 3)
    if success:
        ## make separate folder for video images
        path = os.path.join(baseExtractpath, pathlib.Path(file.rsplit('.', 1)[0]).name + '_' + str(earcounts[count]))
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        imgs.append(resized)
        imshape = resized.shape

        while success:
          success,image = cap.read()
          if success:
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            imgs.append(resized)
          # print('Read a new frame: ', success)

        imgs = np.stack(imgs)
        for ind, img in enumerate(imgs):
            cv2.imwrite(os.path.join(path, 'frame' + str(ind) + '.jpg'), img)
        # imgs = imgs.reshape((imgs.shape[0], -1))
    count += 1

##### mode (background)
# img_mode = st.mode(imgs, axis=0)
# bg = img_mode.mode.reshape((imshape)).astype(np.uint8)
# bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
# plt.imshow(img_mode.mode.reshape((imshape)).astype(np.uint8))