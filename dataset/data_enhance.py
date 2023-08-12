# enhance the data by adding noise and shifting the data
import pandas as pd
import numpy as np

# Load the dataset
name = "data_relabelled_br"
data = pd.read_csv("./dataset/" + name + ".csv")
# data.head()
from sklearn.model_selection import train_test_split

# Splitting the data into pre_train, fine_tune, and test datasets
pre_train, temp_data = train_test_split(data, train_size=0.7, stratify=data['strains'], random_state=42)
fine_tune, test = train_test_split(temp_data, test_size=1/3, stratify=temp_data['strains'], random_state=42)

# Checking the distribution of classes in each dataset
pre_train_distribution = pre_train['strains'].value_counts()
fine_tune_distribution = fine_tune['strains'].value_counts()
test_distribution = test['strains'].value_counts()

# pre_train_distribution, fine_tune_distribution, test_distribution

#############################################################################################################

def augment_data(data, num_augmentations):
    """
    Augment the data using random noise and slight shifts.
    
    Parameters:
    - data: Original data
    - num_augmentations: Number of augmented samples per original data
    
    Returns:
    - Augmented data
    """
    augmented_data = []
    for _, row in data.iterrows():
        for _ in range(num_augmentations):
            # Add random gaussian noise
            noise = np.random.normal(0, 1e-8, len(row) - 1)
            new_row = row.values[:-1] + noise
            
            # Slight random shift
            # shift = np.random.randint(-2, 3)  # Shift by -2 to 2 indices
            # new_row = np.roll(new_row, shift)
            
            # Append the class label
            new_row = np.append(new_row, row['strains'])
            
            augmented_data.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_data, columns=data.columns)
    return augmented_df

num_augmentations = int(110000 / len(pre_train)) - 1
# Since we faced a memory issue, we'll perform the augmentation in batches.
# This function will augment data in chunks to avoid memory constraints.

def augment_data_in_batches(data, num_augmentations, batch_size=1000):
    """
    Augment the data in batches.
    
    Parameters:
    - data: Original data
    - num_augmentations: Number of augmented samples per original data
    - batch_size: Number of samples to augment in each batch
    
    Returns:
    - Augmented data
    """
    num_batches = int(np.ceil(len(data) / batch_size))
    augmented_data_batches = []
    
    for i in range(num_batches):
        batch_data = data.iloc[i*batch_size : (i+1)*batch_size]
        augmented_batch = augment_data(batch_data, num_augmentations)
        augmented_data_batches.append(augmented_batch)
    
    augmented_df = pd.concat(augmented_data_batches, axis=0)
    return augmented_df

# Augment the data in batches
augmented_pre_train_batches = augment_data_in_batches(pre_train, num_augmentations, batch_size=100)

# Combine original and augmented data
augmented_data_batches = pd.concat([pre_train, augmented_pre_train_batches], axis=0)

# augmented_data_batches.shape

#############################################################################################################

# Saving the datasets to CSV files
augmented_data_batches.to_csv("./dataset/pre_enhanced_relabelled_br.csv", index=False)
pre_train.to_csv("./dataset/pre_train_relabelled_br.csv", index=False)
fine_tune.to_csv("./dataset/fine_tune_relabelled_br.csv", index=False)
test.to_csv("./dataset/test_relabelled_br.csv", index=False)


#############################################################################################################

