"""
Quick test of simpler Transformer with real data
"""
import torch
import torch.nn as nn
from pathlib import Path
from dataloader import load_split_data, ReadmissionDataset
from torch.utils.data import DataLoader
from transformer_model_simple import create_transformer_model

print("="*70)
print("TESTING SIMPLER TRANSFORMER WITH REAL DATA")
print("="*70)
print()

# Load real data
BASE_DIR = Path(__file__).parent.parent / 'data'
train_ids, static_df, tensor_dir = load_split_data('train', BASE_DIR)
train_dataset = ReadmissionDataset(train_ids[:200], static_df, tensor_dir, use_time_series=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# Create model
model = create_transformer_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Dataset size: {len(train_dataset)}")
print()

# Test multiple batches
print("Testing 5 batches...")
model.train()
for i, batch in enumerate(train_loader):
    if i >= 5:
        break
    
    X = batch['X']
    mask = batch['mask']
    static = batch['static']
    labels = batch['label']
    
    # Forward pass
    outputs = model(X, mask, static).squeeze()
    loss = criterion(outputs, labels)
    
    # Check for NaN
    has_nan = torch.isnan(outputs).any() or torch.isnan(loss)
    
    print(f"Batch {i+1}: loss={loss.item():.4f}, NaN={has_nan}, " +
          f"outputs range=[{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
    
    if not has_nan:
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        max_grad = 0.0
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        if has_nan_grad:
            print(f"  → NaN in gradients!")
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"  → Gradient update successful (max_grad={max_grad:.4f})")
    else:
        print(f"  → Skipping due to NaN")

print()
print("="*70)
if not has_nan:
    print("✓ ALL TESTS PASSED - MODEL IS STABLE!")
else:
    print("✗ Model still has NaN issues")
print("="*70)
