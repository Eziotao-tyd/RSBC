import re
import pickle
import argparse

# read log file
parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str, default="./checkpoints/test.log", help='path to train log')
args = parser.parse_args()
file_path = args.log_path
out_path = file_path.replace(".log", ".pkl")

with open(file_path, 'r') as file:
    content = file.read()

# Split the file content by "Fold" using regular expression
folds = re.split(r'Fold \d+ / \d+', content)[1:]

# Define regular expression patterns for extracting training loss, validation loss, and validation accuracy
train_loss_pattern = re.compile(r'Training Loss: (\d+\.\d+(?:e[+\-]?\d+)?)')
validation_loss_pattern = re.compile(r'Validation Loss: (\d+\.\d+(?:e[+\-]?\d+)?)')
validation_accuracy_pattern = re.compile(r'Validation Accuracy: (\d+\.\d+)')

results = []

# Iterate through each fold, extract and store training loss, validation loss, and validation accuracy
for fold in folds:
    train_losses = [float(match.group(1)) for match in train_loss_pattern.finditer(fold)]
    validation_losses = [float(match.group(1)) for match in validation_loss_pattern.finditer(fold)]
    validation_accuracies = [float(match.group(1)) for match in validation_accuracy_pattern.finditer(fold)]
    
    results.append({
        'Train Loss': train_losses,
        'Validation Loss': validation_losses,
        'Validation Accuracy': validation_accuracies
    })

# step 1: calculate the maximum accuracy in each fold
max_accuracies = [max(fold['Validation Accuracy']) for fold in results]

# step 2: find the fold with the maximum accuracy
max_accuracy = max(max_accuracies)
max_accuracy_folds = [i for i, acc in enumerate(max_accuracies) if acc == max_accuracy]

# step 3: if there are multiple folds with the same maximum accuracy, select the one with the most accuracy values
if len(max_accuracy_folds) > 1:
    counts = [fold['Validation Accuracy'].count(max_accuracy) for i, fold in enumerate(results) if i in max_accuracy_folds]
    max_count = max(counts)
    max_count_folds = [max_accuracy_folds[i] for i, count in enumerate(counts) if count == max_count]

    # if there are still multiple folds, select the one with the smallest index
    selected_fold_index = min(max_count_folds)
else:
    selected_fold_index = max_accuracy_folds[0]

print(selected_fold_index, max_accuracy)

# save the selected fold to a pickle file
with open(out_path, 'wb') as file:
    pickle.dump(results[selected_fold_index], file)
