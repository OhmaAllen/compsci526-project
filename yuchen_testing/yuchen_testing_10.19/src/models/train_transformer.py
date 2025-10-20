"""
Transformer Training Script - Independent
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from datetime import datetime

from dataloader import load_split_data, ReadmissionDataset
from transformer_model import create_transformer_model

# Paths
BASE_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
MODELS_DIR = RESULTS_DIR / 'models' / 'transformer'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 25  # Reduced from 50 for faster training
PATIENCE = 7  # Reduced proportionally
DEVICE = torch.device('cpu')
SEED = 42

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

log("="*70)
log("TRANSFORMER TRAINING")
log("="*70)
log(f"Device: {DEVICE}")
log(f"Random seed: {SEED}")
log("")

# Load data
log("Loading data...")
train_ids, static_df, tensor_dir = load_split_data('train', BASE_DIR)
val_ids, _, _ = load_split_data('val', BASE_DIR)
test_ids, _, _ = load_split_data('test', BASE_DIR)
log(f"✓ Train: {len(train_ids)} IDs")
log(f"✓ Val: {len(val_ids)} IDs")
log(f"✓ Test: {len(test_ids)} IDs")
log("")

# Create datasets
log("Creating datasets...")
train_dataset = ReadmissionDataset(train_ids, static_df, tensor_dir, use_time_series=True)
val_dataset = ReadmissionDataset(val_ids, static_df, tensor_dir, use_time_series=True)
test_dataset = ReadmissionDataset(test_ids, static_df, tensor_dir, use_time_series=True)
log("✓ Datasets created")
log("")

# Create dataloaders
log("Creating dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
log(f"✓ Train: {len(train_loader)} batches")
log(f"✓ Val: {len(val_loader)} batches")
log(f"✓ Test: {len(test_loader)} batches")
log("")

# Create model
log("Creating Transformer model...")
model = create_transformer_model().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
log(f"✓ Parameters: {total_params:,}")
log("")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        X = batch['X'].to(device)
        mask = batch['mask'].to(device)
        static = batch['static'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(X, mask, static).squeeze()
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            log(f"  WARNING: NaN loss detected, skipping batch")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        if not np.isnan(probs).any():
            total_loss += loss.item()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    if len(all_preds) > 0:
        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
    else:
        auroc, auprc = 0.0, 0.0
    
    return avg_loss, auroc, auprc

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            X = batch['X'].to(device)
            mask = batch['mask'].to(device)
            static = batch['static'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(X, mask, static).squeeze()
            loss = criterion(outputs, labels)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            total_loss += loss.item()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    return avg_loss, auroc, auprc, all_preds, all_labels

# Training loop
log("="*70)
log("STARTING TRAINING")
log("="*70)
log("")

best_val_auroc = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_auroc': [], 'val_loss': [], 'val_auroc': []}

for epoch in range(1, MAX_EPOCHS + 1):
    log(f"Epoch {epoch}/{MAX_EPOCHS}")
    
    # Train
    train_loss, train_auroc, train_auprc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Validate
    val_loss, val_auroc, val_auprc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    
    # Log results
    log(f"  Train - Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f}")
    log(f"  Val   - Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_auroc'].append(train_auroc)
    history['val_loss'].append(val_loss)
    history['val_auroc'].append(val_auroc)
    
    # Save best model
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        torch.save(model.state_dict(), MODELS_DIR / 'transformer_best.pt')
        log(f"  ✓ New best model saved (AUROC: {val_auroc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            log(f"  Early stopping triggered (patience: {PATIENCE})")
            break
    
    # Learning rate scheduling
    scheduler.step(val_auroc)
    log("")

# Load best model and test
log("="*70)
log("TESTING BEST MODEL")
log("="*70)
log("")

model.load_state_dict(torch.load(MODELS_DIR / 'transformer_best.pt'))
test_loss, test_auroc, test_auprc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)

log(f"Test Results:")
log(f"  Loss: {test_loss:.4f}")
log(f"  AUROC: {test_auroc:.4f}")
log(f"  AUPRC: {test_auprc:.4f}")
log("")

# Save results
results = {
    'model': 'Transformer',
    'test_loss': float(test_loss),
    'test_auroc': float(test_auroc),
    'test_auprc': float(test_auprc),
    'best_val_auroc': float(best_val_auroc),
    'total_params': total_params,
    'history': history
}

with open(MODELS_DIR / 'transformer_results.json', 'w') as f:
    json.dump(results, f, indent=2)

log("="*70)
log("✓ TRAINING COMPLETE!")
log(f"✓ Results saved to: {MODELS_DIR / 'transformer_results.json'}")
log("="*70)
