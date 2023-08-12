import sys, os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold

# Import the model and config
from models.transformer_raw import TransformerModel
from utils.config import parse_args
# Load the data using Datareader
from utils.datareader import Datareader

def run(args):
    # Check for GPU availability
    assert torch.cuda.is_available(), "No GPU found!"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"dataset: {args.data_path}")
    print(f"Using Model: {args.model}")
    print(f"Number of folds: {args.num_fold}")

    dataset = Datareader(args.data_path)  # Load entire dataset from test_path

    # 5-fold Cross Validation
    kf = KFold(n_splits=args.num_fold, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1} / {args.num_fold}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        # Define the model, criterion, and optimizer for each fold
        if args.model == "T":
            model = TransformerModel(
                input_dim=args.input_dim, 
                embed_dim=args.embed_dim, 
                num_classes=args.num_classes, 
                num_heads=args.num_heads, 
                num_encoder_layers=args.num_encoder_layers, 
                dropout=args.dropout
            ).to(device)
        else:
            raise ValueError("Invalid model type!")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
        

        # Training loop
        for epoch in tqdm(range(args.train_epochs)):
            model.train()
            total_loss = 0.0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                
            
            print(f"\nEpoch {epoch+1}/{args.train_epochs}, Training Loss: {total_loss/len(train_loader)}")

            # Validation after each epoch
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)

                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)

            print(f"Validation Loss: {val_loss}, Validation Accuracy: {100 * correct / total}%")
            print(f"validation correct/total : {correct}/{total}")
        

if __name__ == "__main__":
    args = parse_args()
    run(args)