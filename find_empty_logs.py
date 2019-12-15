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
bg=np.load(r'D:\AlmacoEarCounts\bg.npy')
for imgs in data_:
    imgs[0] = imgs[0] - bg

# data_split_ind = np.random.permutation(len(data))
# data_loader_train = data[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
# data_loader_val = data[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
# data_loader_test = data[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]

empty_runs = []
for run in data_:
    sig = np.array([np.std(x) for x in run[0]])
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

    empty_runs.append(c) ## c == 0 is empty log
    # empty_runs.append(run[0][c == 0])
    ## plot runs and show empty parts
    # plt.figure();plt.scatter(range(len(sig)), sig, c=c); #plt.figure();
    # plt.scatter(range(len(sig)), sig_filt, c=c)
    #
    # plt.figure();plt.scatter(range(len(sig_filt_diff)), sig_filt_diff, c=c); #plt.figure();