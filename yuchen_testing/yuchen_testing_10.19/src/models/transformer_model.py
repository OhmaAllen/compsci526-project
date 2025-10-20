"""
Transformer Model for 30-Day Readmission Prediction

Architecture:
- Input: Time-series (48, 11) + mask (48, 11) + static features (6)
- Positional encoding for temporal information
- Transformer encoder with multi-head attention
- Concatenate with static features
- Fully connected layers for binary classification

Key features:
- Attention mechanism learns which timepoints are critical
- Mask-aware attention (padding mask)
- Parallel processing of all timepoints
- Better capture of long-range dependencies than LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    
    Adds position information to each timepoint in the sequence.
    """
    
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ReadmissionTransformer(nn.Module):
    """
    Complete Transformer model for readmission prediction.
    
    Architecture:
        Time-series → Embed → Positional Encoding → Transformer Encoder
                                                            ↓
                                                        Pool (mean)
                                                            ↓
        Static features ─────────────────────→ Concat → FC layers → Sigmoid → Prediction
    """
    
    def __init__(
        self,
        time_series_input_size=11,
        static_input_size=6,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        fc_hidden_size=64,
        fc_dropout=0.3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.time_series_input_size = time_series_input_size
        
        # Input embedding: project from input_size to d_model
        self.input_embedding = nn.Linear(time_series_input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=50, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Fully connected layers
        combined_size = d_model + static_input_size
        
        self.fc = nn.Sequential(
            nn.Linear(combined_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, 1)
        )
        
    def create_padding_mask(self, mask):
        """
        Create padding mask for transformer from measurement mask.
        
        Args:
            mask: (batch, seq_len, input_size) - 1 for observed, 0 for missing
            
        Returns:
            padding_mask: (batch, seq_len) - True for padding (all features missing)
        """
        # A timepoint is "padding" if ALL features are missing
        # mask.sum(dim=-1) = 0 means all features missing at that timepoint
        padding_mask = (mask.sum(dim=-1) == 0)  # (batch, seq_len)
        return padding_mask
        
    def forward(self, x_time_series, mask, x_static):
        """
        Args:
            x_time_series: (batch, 48, 11) - Time-series lab values
            mask: (batch, 48, 11) - Mask for observed values
            x_static: (batch, 6) - Static features
            
        Returns:
            output: (batch, 1) - Readmission probability (logits)
            attention_weights: Optional, for interpretability
        """
        batch_size, seq_len, _ = x_time_series.shape
        
        # Embed input to d_model dimensions
        x_embedded = self.input_embedding(x_time_series)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x_encoded = self.pos_encoder(x_embedded)
        
        # Create padding mask (timepoints with all features missing)
        padding_mask = self.create_padding_mask(mask)  # (batch, seq_len)
        
        # Pass through transformer encoder
        # Note: src_key_padding_mask marks positions to ignore
        transformer_output = self.transformer_encoder(
            x_encoded,
            src_key_padding_mask=padding_mask
        )  # (batch, seq_len, d_model)
        
        # Pool over time dimension (mean pooling, ignoring padded positions)
        # Create mask for pooling (invert padding mask)
        pooling_mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        
        # Masked mean pooling with better numerical stability
        masked_output = transformer_output * pooling_mask
        mask_sum = pooling_mask.sum(dim=1).clamp(min=1.0)  # At least 1 to avoid division by zero
        pooled = masked_output.sum(dim=1) / mask_sum  # (batch, d_model)
        
        # Concatenate with static features
        combined = torch.cat([pooled, x_static], dim=1)
        
        # Fully connected layers
        output = self.fc(combined)
        
        return output


def create_transformer_model(config=None):
    """
    Factory function to create Transformer model with default or custom config.
    
    Args:
        config: Dictionary with model hyperparameters (optional)
        
    Returns:
        model: ReadmissionTransformer instance
    """
    default_config = {
        'time_series_input_size': 11,
        'static_input_size': 6,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'fc_hidden_size': 64,
        'fc_dropout': 0.3
    }
    
    if config is not None:
        default_config.update(config)
    
    model = ReadmissionTransformer(**default_config)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Transformer Model...")
    print("="*60)
    
    # Create model
    model = create_transformer_model()
    print(f"Model created:")
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Test forward pass
    batch_size = 32
    seq_len = 48
    time_series_size = 11
    static_size = 6
    
    # Create dummy data
    x_time = torch.randn(batch_size, seq_len, time_series_size)
    mask = torch.randint(0, 2, (batch_size, seq_len, time_series_size)).float()
    x_static = torch.randn(batch_size, static_size)
    
    print(f"Input shapes:")
    print(f"  Time-series: {x_time.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Static: {x_static.shape}")
    print()
    
    # Forward pass
    output = model(x_time, mask, x_static)
    print(f"Output shape: {output.shape}")
    print(f"Output (logits) sample: {output[:5].squeeze()}")
    print()
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(output)
    print(f"Probabilities sample: {probs[:5].squeeze()}")
    print()
    
    print("="*60)
    print("✓ Transformer model test passed!")
    print("="*60)
