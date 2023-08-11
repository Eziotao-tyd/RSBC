import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Datareader(Dataset):
    def __init__(self, filepath):
        """
        Initialize the Datareader by reading a CSV file.
        
        Parameters:
        - filepath: Path to the CSV file.
        """
        dataframe = pd.read_csv(filepath)
        self.data = dataframe.iloc[:, :-1].values
        self.labels = dataframe['strains'].values
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Fetch the data and label at the specified index.
        
        Parameters:
        - index: The index of the data to fetch.
        
        Returns:
        - A tuple containing the data and label.
        """
        data_sample = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return data_sample, label

# Example usage:
# dataset = Datareader('path_to_your_file.csv')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# for data, labels in dataloader:
#     # Your training loop here
