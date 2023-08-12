# Baseline Removal
import sys, os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import signal
from scipy.sparse import eye, diags, csc_matrix
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from BaselineRemoval import BaselineRemoval

filename = "data_2_relabelled"
filepath = "../dataset/" + filename + ".csv"
outpath = "../dataset/" + filename + "_br.csv"

dataframe = pd.read_csv(filepath)
feature_data = dataframe.iloc[:, :-1].values
for i in range(feature_data.shape[1]):
    baseline = BaselineRemoval(feature_data[:, i])
    baseline = baseline.ZhangFit()
    feature_data[:, i] = baseline

# 将其保存为文件
dataframe.iloc[:, :-1] = feature_data
dataframe.to_csv(outpath, index=False)
