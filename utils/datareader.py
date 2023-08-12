import sys, os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from BaselineRemoval import BaselineRemoval

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData

def z_score(data):
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    return data


# Savitzky-Golay for smoothing with general parameters
def SG(data, window_length, polyorder):
    return np.array([signal.savgol_filter(item, window_length, polyorder) for item in data])

class Datareader(Dataset):
    def __init__(self, filepath 
                 #,n_components=0.95 
                 #,airpls_lambda=50, airpls_porder=1, airpls_itermax=30 
                 #,sg_window_length=51, sg_polyorder=5
                 ):
        dataframe = pd.read_csv(filepath)
        feature_data = dataframe.iloc[:, :-1].values
        
        # Applying z-score normalization
        feature_data = z_score(feature_data)
        
        # BaseLineRemoval

        # Applying BaseLineRemoval
        # feature_data : [n_samples, n_features]
        for i in range(feature_data.shape[1]):
            baseline = BaselineRemoval(feature_data[:, i])
            baseline = baseline.ZhangFit()
            feature_data[:, i] = baseline

        # Savitzky-Golay

        # Applying Savitzky-Golay smoothing filter with specified parameters
        # feature_data = SG(feature_data, sg_window_length, sg_polyorder)
        
        # Compute positional encoding
        d_model = feature_data.shape[1]
        sequence_length = feature_data.shape[0]
        wavelengths = dataframe.columns[:-1].astype(float).to_numpy()
        wavelength_diffs = np.diff(wavelengths, prepend=wavelengths[0])
        div_term = wavelength_diffs[np.newaxis, :] * -(np.log(10000.0) / d_model)
        position = np.arange(sequence_length)[:, np.newaxis]
        positional_encoding = np.zeros((sequence_length, d_model))
        positional_encoding[:, 0::2] = np.sin(position * div_term[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(position * div_term[:, 1::2])

        feature = feature_data + positional_encoding

        # Applying PCA for dimensionality reduction if n_components is provided
        # if n_components is not None:
        #     pca = PCA(n_components=n_components)
        #     feature = pca.fit_transform(feature)

        self.data = feature
        
        self.labels = dataframe.iloc[:, -1].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_sample = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return data_sample, label
