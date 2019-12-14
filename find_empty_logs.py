import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture

class parser_:
    pass

args = parser_()
args.p_val = 0.2

data = np.array(np.load(r'D:\AlmacoEarCounts\almaco_earcount_labeled_data.npy', allow_pickle=True))
data_ = np.array(data)
data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
data_loader_val = data[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
data_loader_test = data[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]


bg=np.load(r'D:\AlmacoEarCounts\bg.npy')
for imgs in data_loader_train:
    imgs[0] = imgs[0] - bg
for imgs in data_loader_val:
    imgs[0] = imgs[0] - bg
for imgs in data_loader_test:
    imgs[0] = imgs[0] - bg

sig = np.array([np.std(x) for x in data_loader_train[0][0]])
sig_filt = pd.Series(sig).rolling(8,center=True).median().values
# Fill in NaN's...
mask = np.isnan(sig_filt)
sig_filt[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), sig_filt[~mask])
sig_filt_diff = sig_filt[4:] - sig_filt[:-4]
sig_filt_diff = np.pad(sig_filt_diff, 2, mode='edge')

gmm = mixture.GaussianMixture(2)
gmm.fit(sig_filt.reshape(-1,1))
if np.argmin(gmm.means_):
    c=(gmm.predict(sig_filt.reshape(-1,1)) == 0) + (np.abs(sig_filt_diff) > 0.01)
else:
    c = gmm.predict(sig_filt.reshape(-1, 1)) + (np.abs(sig_filt_diff) > 0.01)

plt.figure();plt.scatter(range(len(sig)), sig, c=c); #plt.figure();
plt.scatter(range(len(sig)), sig_filt, c=c)

plt.figure();plt.scatter(range(len(sig_filt_diff)), sig_filt_diff, c=c); #plt.figure();



