#!/usr/bin/env python3
"""
Simplified Training Script for LSTM and Transformer
With proper logging and monitoring
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from datetime import datetime
from tqdm import tqdm

from lstm_model import create_lstm_model
from transformer_model import create_transformer_model
from dataloader import load_split_data, ReadmissionDataset

# Setup logging
log_file = open('/tmp/training_progress.log', 'w', buffering=1)
def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_file.write(full_msg + "\n")
    log_file.flush()

# Config
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

log("="*70)
log("DEEP LEARNING TRAINING - LSTM & Transformer")
log("="*70)
log(f"Device: {DEVICE}")
log(f"Random seed: {RANDOM_SEED}")
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
        auroc, auprc = 0.5, 0.0
    
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
            if not np.isnan(probs).any() and not torch.isnan(loss):
                total_loss += loss.item()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    if len(all_preds) > 0:
        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
    else:
        auroc, auprc = 0.5, 0.0
    
    return avg_loss, auroc, auprc, all_preds, all_labels

def train_model(model_name, model, train_loader, val_loader, test_loader, device):
    """Train a model with early stopping"""
    log("")
    log("="*70)
    log(f"Training {model_name}")
    log("="*70)
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_auroc = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        log(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_auroc, train_auprc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_auroc, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_auroc)
        
        log(f"  Train - Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f}")
        log(f"  Val   - Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")
        
        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            # Save model
            save_path = MODELS_DIR / f"{model_name.lower()}_best.pt"
            torch.save(model.state_dict(), save_path)
            log(f"  ✓ New best model saved (AUROC: {val_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log(f"  Early stopping triggered (patience: {PATIENCE})")
                break
    
    # Load best model and evaluate on test set
    log(f"\nLoading best {model_name} model for test evaluation...")
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name.lower()}_best.pt"))
    test_loss, test_auroc, test_auprc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    log(f"\n{model_name} Test Results:")
    log(f"  Loss: {test_loss:.4f}")
    log(f"  AUROC: {test_auroc:.4f}")
    log(f"  AUPRC: {test_auprc:.4f}")
    
    return {
        'model': model_name,
        'test_loss': test_loss,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'best_val_auroc': best_val_auroc
    }

# Train LSTM
log("\n" + "="*70)
log("STARTING LSTM TRAINING")
log("="*70)
lstm_model = create_lstm_model()
log(f"LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
lstm_results = train_model('LSTM', lstm_model, train_loader, val_loader, test_loader, DEVICE)

# Train Transformer
log("\n" + "="*70)
log("STARTING TRANSFORMER TRAINING")
log("="*70)
transformer_model = create_transformer_model()
log(f"Transformer parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
transformer_results = train_model('Transformer', transformer_model, train_loader, val_loader, test_loader, DEVICE)

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'device': str(DEVICE)
    },
    'lstm': lstm_results,
    'transformer': transformer_results
}

results_file = RESULTS_DIR / 'deep_learning_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

log("\n" + "="*70)
log("TRAINING COMPLETE!")
log("="*70)
log(f"Results saved to: {results_file}")
log(f"LSTM AUROC: {lstm_results['test_auroc']:.4f}")
log(f"Transformer AUROC: {transformer_results['test_auroc']:.4f}")
log("")

log_file.close()
