"""
1D-CNN + GRU Architecture for HCP Sequence Modeling.
"""
import torch
import torch.nn as nn

class HCPSequenceModel(nn.Module):
    """
    Hybrid 1D-CNN and GRU model for sequence classification.
    Input shape expected: (Batch, Sequence Length, Features) -> (N, 86, F)
    """
    def __init__(self, input_features: int, hidden_size: int, num_classes: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        # 1D-CNN feature extractor (Note: Conv1d expects (Batch, Features, SeqLen))
        self.cnn = nn.Conv1d(in_channels=input_features, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # GRU sequence model
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        # Fully connected classifier
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape: (Batch, Sequence, Features)
        """
        # Conv1d expects (Batch, Channels, Length), so we permute
        x = x.permute(0, 2, 1)
        
        # CNN forward
        x = self.cnn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Revert permutation for GRU: (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        
        # GRU forward
        gru_out, _ = self.gru(x)
        
        # Take the output of the last time step
        last_step_out = gru_out[:, -1, :]
        
        # Classifier forward
        logits = self.fc(last_step_out)
        return logits
