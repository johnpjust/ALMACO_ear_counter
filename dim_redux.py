import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import decomposition
from pandas.plotting import scatter_matrix
import pandas as pd

svdnum = 100

empty_logs=np.load(r'D:\AlmacoEarCounts\empty_log_indx.npy', allow_pickle=True)
empty_logs = np.concatenate(empty_logs, axis=0)

data = np.array(np.load(r'D:\AlmacoEarCounts\almaco_earcount_labeled_data.npy', allow_pickle=True))
data = np.vstack([x[0] for x in data])
data = data.reshape((data.shape[0], -1))
bg=np.load(r'D:\AlmacoEarCounts\bg.npy').reshape((1,-1))
data = data-bg

# _, _, vh = linalg.svd(data, full_matrices=False)
#
# vh = vh[:,:svdnum]

svd = decomposition.TruncatedSVD(n_components=10, n_iter=7, random_state=42)
svd.fit(data)

xfm = svd.transform(data)
# plt.figure();plt.scatter(xfm[:,0], xfm[:,1], c=empty_logs==0)
# plt.figure();plt.scatter(xfm[:,0], xfm[:,1])

plt.figure();plt.scatter(xfm[empty_logs==0,0], xfm[empty_logs==0,1], alpha=0.1)
plt.scatter(xfm[empty_logs==1,0], xfm[empty_logs==1,1], alpha=0.1)

df = pd.DataFrame(xfm)
scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde', c=empty_logs==0)