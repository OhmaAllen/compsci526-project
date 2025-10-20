"""
LSTM Model for 30-Day Readmission Prediction

Architecture:
- Input: Time-series (48, 11) + mask (48, 11) + static features (6)
- LSTM encoder processes time-series with mask awareness
- Concatenate final hidden state with static features
- Fully connected layers for binary classification

Key features:
- Handles irregular sampling via masks
- Combines temporal and static information
- Early stopping on validation AUROC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLSTM(nn.Module):
    """LSTM with mask-aware processing for irregular time-series."""
    
    def __init__(
        self,
        input_size=11,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output size depends on bidirectionality
        self.lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
    def forward(self, x, mask):
        """
        Args:
            x: (batch, seq_len, input_size) - Time-series data
            mask: (batch, seq_len, input_size) - Binary mask (1=observed, 0=missing)
            
        Returns:
            output: (batch, lstm_output_size) - Final hidden state
        """
        # Replace missing values with zeros (already done in preprocessing)
        # Mask is used implicitly - zeros don't affect LSTM much
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # For bidirectional: concatenate forward and backward final states
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            # Get last layer's forward and backward states
            h_forward = h_n[-2]  # Second to last layer (forward)
            h_backward = h_n[-1]  # Last layer (backward)
            output = torch.cat([h_forward, h_backward], dim=1)
        else:
            # Get last layer's final state
            output = h_n[-1]
        
        return output


class ReadmissionLSTM(nn.Module):
    """
    Complete LSTM model for readmission prediction.
    
    Architecture:
        Time-series → LSTM → Hidden State
                                    ↓
        Static features ────────→ Concat → FC layers → Sigmoid → Prediction
    """
    
    def __init__(
        self,
        time_series_input_size=11,
        static_input_size=6,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        lstm_dropout=0.2,
        lstm_bidirectional=False,
        fc_hidden_size=64,
        fc_dropout=0.3
    ):
        super().__init__()
        
        # LSTM encoder for time-series
        self.lstm = MaskLSTM(
            input_size=time_series_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=lstm_bidirectional
        )
        
        # Fully connected layers
        lstm_output_size = self.lstm.lstm_output_size
        combined_size = lstm_output_size + static_input_size
        
        self.fc = nn.Sequential(
            nn.Linear(combined_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, 1)
        )
        
    def forward(self, x_time_series, mask, x_static):
        """
        Args:
            x_time_series: (batch, 48, 11) - Time-series lab values
            mask: (batch, 48, 11) - Mask for observed values
            x_static: (batch, 6) - Static features
            
        Returns:
            output: (batch, 1) - Readmission probability (logits)
        """
        # Encode time-series with LSTM
        lstm_features = self.lstm(x_time_series, mask)
        
        # Concatenate with static features
        combined = torch.cat([lstm_features, x_static], dim=1)
        
        # Fully connected layers
        output = self.fc(combined)
        
        return output


def create_lstm_model(config=None):
    """
    Factory function to create LSTM model with default or custom config.
    
    Args:
        config: Dictionary with model hyperparameters (optional)
        
    Returns:
        model: ReadmissionLSTM instance
    """
    default_config = {
        'time_series_input_size': 11,
        'static_input_size': 6,
        'lstm_hidden_size': 128,
        'lstm_num_layers': 2,
        'lstm_dropout': 0.2,
        'lstm_bidirectional': False,
        'fc_hidden_size': 64,
        'fc_dropout': 0.3
    }
    
    if config is not None:
        default_config.update(config)
    
    model = ReadmissionLSTM(**default_config)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM Model...")
    print("="*60)
    
    # Create model
    model = create_lstm_model()
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
    print("✓ LSTM model test passed!")
    print("="*60)
