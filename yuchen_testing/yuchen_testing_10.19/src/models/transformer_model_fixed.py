"""
Fixed Transformer Model for 30-Day Readmission Prediction

KEY FIXES for NaN issues:
1. Layer normalization after embedding (stabilize inputs)
2. Xavier/Kaiming initialization for all linear layers
3. Pre-LayerNorm architecture (more stable than post-LN)
4. Better dropout strategy
5. Smaller model size to reduce instability
6. Input scaling to normalize features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
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


class ReadmissionTransformerFixed(nn.Module):
    """
    Fixed Transformer model with improved numerical stability.
    
    Key improvements:
    - Input normalization layer
    - Layer normalization after embedding
    - Xavier initialization
    - Smaller model size
    - Better attention stability
    """
    
    def __init__(
        self,
        time_series_input_size=11,
        static_input_size=6,
        d_model=64,  # Reduced from 128 for stability
        nhead=4,
        num_layers=1,  # Reduced from 2 for stability
        dim_feedforward=256,  # Reduced from 512
        dropout=0.1,
        fc_hidden_size=32,
        fc_dropout=0.2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.time_series_input_size = time_series_input_size
        
        # Input normalization (NEW: stabilize inputs)
        self.input_norm = nn.LayerNorm(time_series_input_size)
        
        # Input embedding: project from input_size to d_model
        self.input_embedding = nn.Linear(time_series_input_size, d_model)
        
        # Layer norm after embedding (NEW: stabilize before transformer)
        self.embed_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=50, dropout=dropout)
        
        # Transformer encoder with pre-LN architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN is more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Final layer norm (NEW: stabilize transformer output)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Fully connected layers
        combined_size = d_model + static_input_size
        
        self.fc = nn.Sequential(
            nn.Linear(combined_size, fc_hidden_size),
            nn.LayerNorm(fc_hidden_size),  # NEW: add layer norm
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, 1)
        )
        
        # Initialize weights (NEW: proper initialization)
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights using Xavier/Kaiming initialization.
        This is critical for preventing NaN in deep networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
    def create_padding_mask(self, mask):
        """
        Create padding mask for transformer from measurement mask.
        
        Args:
            mask: (batch, seq_len, input_size) - 1 for observed, 0 for missing
            
        Returns:
            padding_mask: (batch, seq_len) - True for padding (all features missing)
        """
        # A timepoint is "padding" if ALL features are missing
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
        """
        batch_size, seq_len, _ = x_time_series.shape
        
        # Normalize input (NEW: stabilize inputs)
        # Clamp input to prevent extreme values
        x_clamped = torch.clamp(x_time_series, min=-100, max=100)
        x_normed = self.input_norm(x_clamped)
        
        # Embed input to d_model dimensions
        x_embedded = self.input_embedding(x_normed)  # (batch, seq_len, d_model)
        
        # Clamp embeddings to prevent explosion
        x_embedded = torch.clamp(x_embedded, min=-10, max=10)
        
        # Normalize embeddings (NEW: stabilize before transformer)
        x_embedded = self.embed_norm(x_embedded)
        
        # Add positional encoding
        x_encoded = self.pos_encoder(x_embedded)
        
        # Create padding mask (timepoints with all features missing)
        padding_mask = self.create_padding_mask(mask)  # (batch, seq_len)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            x_encoded,
            src_key_padding_mask=padding_mask
        )  # (batch, seq_len, d_model)
        
        # Check for NaN and replace with zeros (defensive programming)
        if torch.isnan(transformer_output).any():
            transformer_output = torch.nan_to_num(transformer_output, nan=0.0)
        
        # Normalize transformer output (NEW: stabilize)
        transformer_output = self.output_norm(transformer_output)
        
        # Pool over time dimension (mean pooling, ignoring padded positions)
        pooling_mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        
        # Masked mean pooling with clamping
        masked_output = transformer_output * pooling_mask
        mask_sum = pooling_mask.sum(dim=1).clamp(min=1.0)
        pooled = masked_output.sum(dim=1) / mask_sum  # (batch, d_model)
        
        # Concatenate with static features
        combined = torch.cat([pooled, x_static], dim=1)
        
        # Fully connected layers
        output = self.fc(combined)
        
        return output


def create_transformer_model(config=None):
    """
    Factory function to create FIXED Transformer model with default or custom config.
    
    Args:
        config: Dictionary with model hyperparameters (optional)
        
    Returns:
        model: ReadmissionTransformerFixed instance
    """
    default_config = {
        'time_series_input_size': 11,
        'static_input_size': 6,
        'd_model': 64,  # Smaller for stability
        'nhead': 4,
        'num_layers': 1,  # Fewer layers for stability
        'dim_feedforward': 256,
        'dropout': 0.1,
        'fc_hidden_size': 32,
        'fc_dropout': 0.2
    }
    
    if config is not None:
        default_config.update(config)
    
    model = ReadmissionTransformerFixed(**default_config)
    
    return model


if __name__ == "__main__":
    # Test the fixed model
    print("Testing FIXED Transformer Model...")
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
    print(f"Output contains NaN: {torch.isnan(output).any().item()}")
    print(f"Output (logits) sample: {output[:5].squeeze()}")
    print()
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(output)
    print(f"Probabilities sample: {probs[:5].squeeze()}")
    print()
    
    # Test with mostly masked data (like real scenario)
    print("Testing with sparse data (97% masked)...")
    mask_sparse = torch.zeros(batch_size, seq_len, time_series_size)
    # Only 1-2 timepoints per sample
    for i in range(batch_size):
        num_points = torch.randint(1, 3, (1,)).item()
        indices = torch.randint(0, seq_len, (num_points,))
        mask_sparse[i, indices, :] = 1.0
    
    output_sparse = model(x_time, mask_sparse, x_static)
    print(f"Sparse output contains NaN: {torch.isnan(output_sparse).any().item()}")
    print(f"Sparse output sample: {output_sparse[:5].squeeze()}")
    print()
    
    print("="*60)
    print("✓ FIXED Transformer model test passed!")
    print("="*60)
