# Change the label to start from 0
import pandas as pd
import argparse
# Load the data

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./dataset/data_2.csv", help='path to raw data')
args = parser.parse_args()

data_path = args.data_path
#f"./dataset/data.csv"
data = pd.read_csv(data_path)

# Discretize and re-label the 'strains' column
data['strains'] = data['strains'].astype('category').cat.codes

# Rename the 'Labels' column to 'strains'
# data = data.rename(columns={'Labels': 'strains'})


transformed_data_path = data_path.replace(".csv", "_relabelled.csv")
data.to_csv(transformed_data_path, index=False)