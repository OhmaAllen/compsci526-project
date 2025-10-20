#!/usr/bin/env python3
"""
Train LSTM and Transformer Models for Readmission Prediction

Trains both deep learning models with:
- Early stopping on validation AUROC
- Learning rate scheduling
- Model checkpointing
- Comprehensive evaluation

Models:
1. LSTM: Sequential processing of time-series
2. Transformer: Attention-based temporal modeling
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

# Import models
from lstm_model import create_lstm_model
from transformer_model import create_transformer_model
from dataloader import load_split_data, ReadmissionDataset

# Set random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
MODELS_DIR = RESULTS_DIR / 'models'

# Training config
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping patience
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Force unbuffered output and logging
import sys
import builtins
_log_file = open('/tmp/deep_learning_training.log', 'w', buffering=1)
_original_print = builtins.print

def log_print(msg="", **kwargs):
    """Print to both console and log file"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_msg = f"[{timestamp}] {msg}" if msg else ""
    _original_print(full_msg, flush=True, **kwargs)
    _log_file.write(full_msg + "\n")
    _log_file.flush()

# Replace print for this script
builtins.print = log_print

print("="*70)
print("DEEP LEARNING MODELS: LSTM & Transformer")
print("="*70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"Random seed: {RANDOM_SEED}")
print()


# ============================================================================
# Helper Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move to device
        X = batch['X'].to(device)
        mask = batch['mask'].to(device)
        static = batch['static'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X, mask, static).squeeze()
        loss = criterion(outputs, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            print("WARNING: NaN loss detected, skipping batch")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Collect predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        
        # Check for NaN in predictions
        if not np.isnan(probs).any():
            total_loss += loss.item()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Check for empty or NaN predictions
    if len(all_preds) == 0:
        print("WARNING: No valid predictions collected in training")
        return 0.0, 0.5, 0.0
    
    # Remove any NaN values
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    valid_mask = ~(np.isnan(all_preds) | np.isinf(all_preds))
    
    if valid_mask.sum() == 0:
        print("WARNING: All predictions are NaN/inf")
        return avg_loss, 0.5, 0.0
    
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    return avg_loss, auroc, auprc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move to device
            X = batch['X'].to(device)
            mask = batch['mask'].to(device)
            static = batch['static'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(X, mask, static).squeeze()
            loss = criterion(outputs, labels)
            
            # Collect predictions
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Check for NaN
            if not np.isnan(probs).any() and not torch.isnan(loss):
                total_loss += loss.item()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Check for empty or NaN predictions
    if len(all_preds) == 0:
        print("WARNING: No valid predictions collected in evaluation")
        return 0.0, 0.5, 0.0, [], []
    
    # Remove any NaN values
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    valid_mask = ~(np.isnan(all_preds) | np.isinf(all_preds))
    
    if valid_mask.sum() == 0:
        print("WARNING: All predictions are NaN/inf in evaluation")
        return avg_loss, 0.5, 0.0, all_preds.tolist(), all_labels.tolist()
    
    # Filter NaN values
    filtered_preds = all_preds[valid_mask]
    filtered_labels = all_labels[valid_mask]
    
    auroc = roc_auc_score(filtered_labels, filtered_preds)
    auprc = average_precision_score(filtered_labels, filtered_preds)
    
    return avg_loss, auroc, auprc, all_preds.tolist(), all_labels.tolist()


def train_model(model_name, model, train_loader, val_loader, test_loader, device):
    """
    Train a model with early stopping.
    
    Args:
        model_name: 'LSTM' or 'Transformer'
        model: PyTorch model
        train_loader, val_loader, test_loader: DataLoaders
        device: Device to train on
        
    Returns:
        results: Dictionary with metrics and predictions
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name} Model")
    print(f"{'='*70}")
    
    # Setup
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_auroc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_auroc': [], 'val_loss': [], 'val_auroc': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*70)
        
        # Train
        train_loss, train_auroc, train_auprc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_auroc, val_auprc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_auroc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_auroc'].append(train_auroc)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val AUROC:   {val_auroc:.4f}")
        
        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            # Save best model
            best_model_path = MODELS_DIR / f'{model_name.lower()}_best.pt'
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! (AUROC: {best_val_auroc:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    print(f"\nLoading best model (Val AUROC: {best_val_auroc:.4f})")
    model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_auroc, test_auprc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*70}")
    print(f"{model_name} FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Val AUROC:  {best_val_auroc:.4f}")
    print(f"Test AUROC:      {test_auroc:.4f}")
    print(f"Test AUPRC:      {test_auprc:.4f}")
    print(f"{'='*70}")
    
    # Return results
    results = {
        'model_name': model_name,
        'best_val_auroc': float(best_val_auroc),
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'test_loss': float(test_loss),
        'history': history,
        'test_predictions': test_preds,
        'test_labels': test_labels
    }
    
    return results


# ============================================================================
# Main Training
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING DATA LOADING")
    print("="*70)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    print("Loading data...")
    print("-"*70)
    
    # Load data
    print("Loading train split...")
    train_ids, static_df, tensor_dir = load_split_data('train', BASE_DIR)
    print(f"✓ Train: {len(train_ids)} IDs, static_df: {static_df.shape}")
    
    print("Loading val split...")
    val_ids, _, _ = load_split_data('val', BASE_DIR)
    print(f"✓ Val: {len(val_ids)} IDs")
    
    print("Loading test split...")
    test_ids, _, _ = load_split_data('test', BASE_DIR)
    print(f"✓ Test: {len(test_ids)} IDs")
    print()
    
    # Create datasets (with time-series)
    print("Creating datasets...")
    train_dataset = ReadmissionDataset(train_ids, static_df, tensor_dir, use_time_series=True)
    val_dataset = ReadmissionDataset(val_ids, static_df, tensor_dir, use_time_series=True)
    test_dataset = ReadmissionDataset(test_ids, static_df, tensor_dir, use_time_series=True)
    print("✓ All datasets created")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("✓ All dataloaders created")
    print()
    
    print(f"Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_dataset):,} samples ({len(val_loader)} batches)")
    print(f"Test:  {len(test_dataset):,} samples ({len(test_loader)} batches)")
    print()
    
    # Create results directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train LSTM
    lstm_model = create_lstm_model()
    lstm_results = train_model('LSTM', lstm_model, train_loader, val_loader, test_loader, DEVICE)
    
    # Train Transformer
    transformer_model = create_transformer_model()
    transformer_results = train_model('Transformer', transformer_model, train_loader, val_loader, test_loader, DEVICE)
    
    # ============================================================================
    # Compare Results
    # ============================================================================
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison = {
        'LSTM': {
            'val_auroc': lstm_results['best_val_auroc'],
            'test_auroc': lstm_results['test_auroc'],
            'test_auprc': lstm_results['test_auprc']
        },
        'Transformer': {
            'val_auroc': transformer_results['best_val_auroc'],
            'test_auroc': transformer_results['test_auroc'],
            'test_auprc': transformer_results['test_auprc']
        }
    }
    
    print("\n{:<15} {:>12} {:>12} {:>12}".format(
        "Model", "Val AUROC", "Test AUROC", "Test AUPRC"
    ))
    print("-"*70)
    for model_name, metrics in comparison.items():
        print("{:<15} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            model_name,
            metrics['val_auroc'],
            metrics['test_auroc'],
            metrics['test_auprc']
        ))
    
    # Best model
    best_model = max(comparison.items(), key=lambda x: x[1]['test_auroc'])
    print(f"\n✓ Best model: {best_model[0]} (Test AUROC: {best_model[1]['test_auroc']:.4f})")
    
    # Save results
    results_file = RESULTS_DIR / 'deep_learning_results.json'
    with open(results_file, 'w') as f:
        # Remove large arrays before saving
        lstm_save = {k: v for k, v in lstm_results.items() if k not in ['test_predictions', 'test_labels']}
        transformer_save = {k: v for k, v in transformer_results.items() if k not in ['test_predictions', 'test_labels']}
        json.dump({'lstm': lstm_save, 'transformer': transformer_save}, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': lstm_results['test_labels'],
        'lstm_proba': lstm_results['test_predictions'],
        'transformer_proba': transformer_results['test_predictions']
    })
    predictions_file = RESULTS_DIR / 'deep_learning_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Predictions saved to {predictions_file}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
