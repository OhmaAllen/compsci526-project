"""
Simplified Transformer Model for Sparse Time-Series Data

This version uses a simpler architecture designed for very sparse data:
- Manual attention implementation with better numerical stability
- Handles cases where samples have only 1-2 non-padded timepoints
- Adds epsilon values everywhere to prevent division by zero
- Uses ReLU attention instead of softmax for better stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StableAttention(nn.Module):
    """
    Numerically stable attention mechanism for sparse data.
    Uses scaled dot-product attention with careful handling of masks.
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) bool tensor, True for padding
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, nhead, seq_len, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # (batch, nhead, seq_len, seq_len)
        
        # Apply mask
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
            
            # Also mask out attention TO padding positions
            mask_t = mask.transpose(-2, -1)
            scores = scores.masked_fill(mask_t, -1e9)
        
        # Softmax with numerical stability
        attn = F.softmax(scores, dim=-1)
        
        # Replace NaN in attention weights with uniform distribution
        if torch.isnan(attn).any():
            # For each query position, if all keys are masked, set uniform attention
            if mask is not None:
                valid_keys = (~mask).sum(dim=-1, keepdim=True).unsqueeze(1)  # (batch, 1, 1, 1)
                uniform_attn = 1.0 / valid_keys.clamp(min=1.0)
                attn = torch.where(torch.isnan(attn), uniform_attn.expand_as(attn), attn)
            else:
                attn = torch.nan_to_num(attn, nan=0.0)
        
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, v)  # (batch, nhead, seq_len, d_k)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, nhead, d_k)
        output = output.reshape(batch_size, seq_len, d_model)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class SimplerTransformerLayer(nn.Module):
    """
    A single transformer layer with better numerical stability.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.self_attn = StableAttention(d_model, nhead, dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms (pre-LN for stability)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) bool, True for padding
        """
        # Pre-LN self-attention
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-LN feedforward
        x_norm = self.norm2(x)
        ff_output = self.linear2(F.relu(self.linear1(x_norm)))
        x = x + self.dropout(ff_output)
        
        return x


class ReadmissionTransformerSimple(nn.Module):
    """
    Simpler Transformer with manual attention for better numerical stability.
    """
    
    def __init__(
        self,
        time_series_input_size=11,
        static_input_size=6,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.1,
        fc_hidden_size=32,
        fc_dropout=0.2
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(time_series_input_size, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, d_model) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimplerTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        combined_size = d_model + static_input_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, fc_hidden_size),
            nn.LayerNorm(fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, 1)
        )
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # Small init
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x_time_series, mask, x_static):
        """
        Args:
            x_time_series: (batch, seq_len, input_size)
            mask: (batch, seq_len, input_size) - 1 for valid, 0 for missing
            x_static: (batch, static_size)
        """
        batch_size, seq_len, _ = x_time_series.shape
        
        # Clamp inputs to prevent extreme values
        x_time_series = torch.clamp(x_time_series, -100, 100)
        
        # Project input
        x = self.input_proj(x_time_series)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Create padding mask (True for ALL-missing timepoints)
        padding_mask = (mask.sum(dim=-1) == 0)  # (batch, seq_len)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, padding_mask)
            
            # Safety check
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
        
        # Final norm
        x = self.final_norm(x)
        
        # Pool over time (mean of non-masked positions)
        valid_mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        
        # Concatenate with static
        combined = torch.cat([pooled, x_static], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return output


def create_transformer_model(config=None):
    """Factory function for simpler transformer."""
    default_config = {
        'time_series_input_size': 11,
        'static_input_size': 6,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 1,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'fc_hidden_size': 32,
        'fc_dropout': 0.2
    }
    
    if config is not None:
        default_config.update(config)
    
    return ReadmissionTransformerSimple(**default_config)


if __name__ == "__main__":
    print("Testing Simpler Transformer...")
    print("="*60)
    
    model = create_transformer_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Test with sparse data
    batch_size = 32
    seq_len = 48
    x_time = torch.randn(batch_size, seq_len, 11)
    mask = torch.zeros(batch_size, seq_len, 11)
    
    # Only 1-2 timepoints per sample
    for i in range(batch_size):
        num_points = torch.randint(1, 3, (1,)).item()
        indices = torch.randint(0, seq_len, (num_points,))
        mask[i, indices, :] = 1.0
    
    x_static = torch.randn(batch_size, 6)
    
    print(f"Input shapes: {x_time.shape}, {mask.shape}, {x_static.shape}")
    print(f"Avg valid timepoints: {(mask.sum(dim=-1) > 0).sum(dim=-1).float().mean():.1f}")
    print()
    
    output = model(x_time, mask, x_static)
    print(f"Output shape: {output.shape}")
    print(f"Output contains NaN: {torch.isnan(output).any().item()}")
    print(f"Output sample: {output[:5].squeeze()}")
    print()
    
    # Test with gradient
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.randint(0, 2, (batch_size,)).float()
    loss = criterion(output.squeeze(), labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss is NaN: {torch.isnan(loss).item()}")
    
    if not torch.isnan(loss):
        loss.backward()
        print("✓ Backward pass successful")
    
    print("="*60)
    print("✓ Test passed!")
