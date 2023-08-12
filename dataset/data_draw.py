# draw the original and corrected spectra for the selected samples
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = "../dataset/data_relabelled.csv"
br_file_path = '../dataset/data_relabelled_br.csv'

# Choose a few samples for comparison
sample_indices = [0, 10, 20]  # Selected arbitrary indices

# Plotting the original and corrected spectra for the selected samples
plt.figure(figsize=(15, 10))


df = pd.read_csv(file_path)
br_df = pd.read_csv(br_file_path)
for i, index in enumerate(sample_indices, start=1):
    plt.subplot(3, 2, i * 2 - 1)
    plt.plot(df.iloc[index, :-1], label='Original', color='blue')
    plt.title(f'Sample {index} Original Spectrum')
    plt.xlabel('Feature')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, i * 2)
    plt.plot(br_df.iloc[index, :-1], label='Corrected', color='red')
    plt.title(f'Sample {index} Corrected Spectrum')
    plt.xlabel('Feature')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('../dataset/br.png')
