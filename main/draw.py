import matplotlib.pyplot as plt
import pickle
import argparse

# path of pickle file
parser = argparse.ArgumentParser()
parser.add_argument('--pkl_path', type=str, default="./checkpoints/test.pkl", help='path to pkl')
parser.add_argument('--out_path', type=str, default="./test.png", help='path to output file')
args = parser.parse_args()
pickle_file_path = args.pkl_path
out_path = args.out_path

# read pickle file
with open(pickle_file_path, 'rb') as file:
    fold_data = pickle.load(file)

# get data
train_losses = fold_data['Train Loss']
validation_losses = fold_data['Validation Loss']
validation_accuracies = fold_data['Validation Accuracy']

# set epochs
epochs = range(1, len(train_losses) + 1)

# get data from 1000 to 2000
# epochs = epochs[1000:2000]
# train_losses = train_losses[1000:2000]
# validation_losses = validation_losses[1000:2000]
# validation_accuracies = validation_accuracies[1000:2000]

# create a new figure
plt.figure(figsize=(15, 5))

# subgraph 1: training loss
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# subgraph 2: validation loss
plt.subplot(1, 3, 2)
plt.plot(epochs, validation_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.grid(True)

# subgraph 3: validation accuracy
plt.subplot(1, 3, 3)
plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.grid(True)

# adjust layout
plt.tight_layout()

# save figure

plt.savefig(out_path)
