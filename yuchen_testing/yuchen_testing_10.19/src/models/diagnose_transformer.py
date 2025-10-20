"""
Diagnostic Script to Identify Transformer Issues

This script systematically tests each component of the Transformer model
to identify where NaN values are being introduced.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataloader import load_split_data, ReadmissionDataset
from torch.utils.data import DataLoader
from transformer_model_simple import create_transformer_model

print("="*70)
print("TRANSFORMER DIAGNOSTIC TEST")
print("="*70)
print()

# Load a small batch of real data
BASE_DIR = Path(__file__).parent.parent / 'data'
train_ids, static_df, tensor_dir = load_split_data('train', BASE_DIR)
train_dataset = ReadmissionDataset(train_ids[:100], static_df, tensor_dir, use_time_series=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

batch = next(iter(train_loader))
X = batch['X']
mask = batch['mask']
static = batch['static']
labels = batch['label']

print("1. INPUT DATA CHECK")
print("-"*70)
print(f"X shape: {X.shape}")
print(f"X contains NaN: {torch.isnan(X).any().item()}")
print(f"X contains Inf: {torch.isinf(X).any().item()}")
print(f"X min: {X.min().item():.4f}, max: {X.max().item():.4f}, mean: {X.mean().item():.4f}")
print(f"Mask sum: {mask.sum().item():.0f} / {mask.numel()} ({100*mask.sum().item()/mask.numel():.1f}%)")
print(f"Static shape: {static.shape}")
print(f"Static contains NaN: {torch.isnan(static).any().item()}")
print(f"Static min: {static.min().item():.4f}, max: {static.max().item():.4f}")
print()

# Create model
model = create_transformer_model()
model.eval()

print("2. INPUT EMBEDDING CHECK")
print("-"*70)
with torch.no_grad():
    x_embedded = model.input_embedding(X)
    print(f"Embedded shape: {x_embedded.shape}")
    print(f"Embedded contains NaN: {torch.isnan(x_embedded).any().item()}")
    print(f"Embedded contains Inf: {torch.isinf(x_embedded).any().item()}")
    print(f"Embedded min: {x_embedded.min().item():.4f}, max: {x_embedded.max().item():.4f}, mean: {x_embedded.mean().item():.4f}")
print()

print("3. POSITIONAL ENCODING CHECK")
print("-"*70)
with torch.no_grad():
    x_encoded = model.pos_encoder(x_embedded)
    print(f"Encoded shape: {x_encoded.shape}")
    print(f"Encoded contains NaN: {torch.isnan(x_encoded).any().item()}")
    print(f"Encoded contains Inf: {torch.isinf(x_encoded).any().item()}")
    print(f"Encoded min: {x_encoded.min().item():.4f}, max: {x_encoded.max().item():.4f}, mean: {x_encoded.mean().item():.4f}")
print()

print("4. PADDING MASK CHECK")
print("-"*70)
padding_mask = model.create_padding_mask(mask)
print(f"Padding mask shape: {padding_mask.shape}")
print(f"Padding mask dtype: {padding_mask.dtype}")
print(f"Padding positions: {padding_mask.sum().item():.0f} / {padding_mask.numel()} ({100*padding_mask.sum().item()/padding_mask.numel():.1f}%)")
print(f"Non-padding positions per sample: {(~padding_mask).sum(dim=1).float().mean().item():.1f}")
print()

print("5. TRANSFORMER ENCODER CHECK")
print("-"*70)
with torch.no_grad():
    try:
        transformer_output = model.transformer_encoder(x_encoded, src_key_padding_mask=padding_mask)
        print(f"Transformer output shape: {transformer_output.shape}")
        print(f"Transformer output contains NaN: {torch.isnan(transformer_output).any().item()}")
        print(f"Transformer output contains Inf: {torch.isinf(transformer_output).any().item()}")
        print(f"Transformer output min: {transformer_output.min().item():.4f}, max: {transformer_output.max().item():.4f}")
        
        # Check attention weights variability
        print(f"Transformer output std: {transformer_output.std().item():.4f}")
        print(f"Transformer output per-sample std: min={transformer_output.std(dim=[1,2]).min().item():.4f}, max={transformer_output.std(dim=[1,2]).max().item():.4f}")
    except Exception as e:
        print(f"ERROR in transformer encoder: {e}")
        transformer_output = None
print()

if transformer_output is not None:
    print("6. POOLING CHECK")
    print("-"*70)
    with torch.no_grad():
        pooling_mask = (~padding_mask).unsqueeze(-1).float()
        print(f"Pooling mask shape: {pooling_mask.shape}")
        print(f"Pooling mask sum per sample: {pooling_mask.sum(dim=1).squeeze()[:5]}")
        
        masked_output = transformer_output * pooling_mask
        print(f"Masked output contains NaN: {torch.isnan(masked_output).any().item()}")
        
        mask_sum = pooling_mask.sum(dim=1).clamp(min=1.0)
        print(f"Mask sum shape: {mask_sum.shape}")
        print(f"Mask sum min: {mask_sum.min().item():.1f}, max: {mask_sum.max().item():.1f}")
        
        pooled = masked_output.sum(dim=1) / mask_sum
        print(f"Pooled shape: {pooled.shape}")
        print(f"Pooled contains NaN: {torch.isnan(pooled).any().item()}")
        print(f"Pooled contains Inf: {torch.isinf(pooled).any().item()}")
        print(f"Pooled min: {pooled.min().item():.4f}, max: {pooled.max().item():.4f}, mean: {pooled.mean().item():.4f}")
    print()

    print("7. CONCATENATION CHECK")
    print("-"*70)
    with torch.no_grad():
        combined = torch.cat([pooled, static], dim=1)
        print(f"Combined shape: {combined.shape}")
        print(f"Combined contains NaN: {torch.isnan(combined).any().item()}")
        print(f"Combined min: {combined.min().item():.4f}, max: {combined.max().item():.4f}")
    print()

    print("8. FC LAYERS CHECK")
    print("-"*70)
    with torch.no_grad():
        output = model.fc(combined)
        print(f"Output shape: {output.shape}")
        print(f"Output contains NaN: {torch.isnan(output).any().item()}")
        print(f"Output contains Inf: {torch.isinf(output).any().item()}")
        print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
        print(f"Output sample: {output[:5].squeeze()}")
    print()

print("9. FULL FORWARD PASS CHECK")
print("-"*70)
with torch.no_grad():
    output = model(X, mask, static)
    print(f"Output shape: {output.shape}")
    print(f"Output contains NaN: {torch.isnan(output).any().item()}")
    print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
    print(f"Output sample: {output[:5].squeeze()}")
print()

print("10. LOSS COMPUTATION CHECK")
print("-"*70)
criterion = nn.BCEWithLogitsLoss()
with torch.no_grad():
    loss = criterion(output.squeeze(), labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss is NaN: {torch.isnan(loss).item()}")
print()

print("11. GRADIENT CHECK (ONE BACKWARD PASS)")
print("-"*70)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
output = model(X, mask, static).squeeze()
loss = criterion(output, labels)
print(f"Training mode - Loss: {loss.item():.4f}")
print(f"Training mode - Loss is NaN: {torch.isnan(loss).item()}")

if not torch.isnan(loss):
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_nan_grad = False
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  NaN gradient in: {name}")
                has_nan_grad = True
            grad_max = param.grad.abs().max().item()
            if grad_max > max_grad:
                max_grad = grad_max
    
    print(f"Has NaN gradients: {has_nan_grad}")
    print(f"Max gradient magnitude: {max_grad:.4f}")
    
    # Clip and update
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print("✓ Optimization step completed")
else:
    print("Cannot compute gradients - loss is NaN")
print()

print("="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
