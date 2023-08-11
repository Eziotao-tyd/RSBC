import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import parse_args

# Load the data using Datareader
from utils.datareader import Datareader

from model.transformer_raw import TransformerModel

def run(Config):
    # Check for GPU availability
    assert torch.cuda.is_available(), "No GPU found!"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    pre_train_dataset = Datareader(args.data_path)
    fine_tune_dataset = Datareader(args.fine_tune_path)
    train_loader = DataLoader(pre_train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
    val_loader = DataLoader(fine_tune_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the model, criterion, and optimizer
    model = TransformerModel(
        input_dim=args.input_dim, 
        embed_dim=args.embed_dim, 
        num_classes=args.num_classes, 
        num_heads=args.num_heads, 
        num_encoder_layers=args.num_encoder_layers, 
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {total_loss/len(train_loader)}")

        # Validation after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print(f"Validation Accuracy: {100 * correct / total}%")

    # Save the pre-trained model
    torch.save(model.state_dict(), './pre_train/pretrained_model.pth')

if __name__ == "__main__":
    args = parse_args()
    run(args)