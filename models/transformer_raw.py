import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_classes, num_heads=4, num_encoder_layers=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)  #(batch_size, embed_dim)
        x = self.classifier(x)  # Use the first token for classification
        return x

# Example usage:
# Assuming input_dim is 467 (the number of features in your data) and you have 9 classes.
# model = TransformerModel(input_dim=467, embed_dim=512, num_classes=9)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Pre-training loop:
# for epoch in range(num_epochs):
#     for data, labels in dataloader:
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
