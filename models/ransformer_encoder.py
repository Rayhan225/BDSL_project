import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super().__init__()
        self.encoding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return x + self.encoding(positions)

class BdSLTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input Projection: 258 dimensions to 128
        self.input_proj = nn.Sequential(
            nn.Linear(258, 128),
            nn.LayerNorm(128)
        )
        
        self.pos_encoder = PositionalEncoding(d_model=128, max_len=30)
        
        # Transformer Encoder: 4 layers, 4 heads, 256 feedforward, 0.1 dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=4, 
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification Head
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 30, 258)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Pool the sequence by taking the mean across the 30 frames
        x = x.mean(dim=1) 
        logits = self.classifier(x)
        return logits