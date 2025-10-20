#!/usr/bin/env python3
"""
Quick test to see where training hangs
"""
import sys
import time
from pathlib import Path
from datetime import datetime

# Setup logging to file
log_file = open('/tmp/training_debug.log', 'w', buffering=1)
def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_file.write(f"[{timestamp}] {msg}\n")
    log_file.flush()
    print(msg, flush=True)

log("="*70)
log("TRAINING DEBUG TEST")
log("="*70)

log("Adding path...")
sys.path.append(str(Path(__file__).parent))

log("Importing torch...")
import torch
log(f"✓ torch version: {torch.__version__}")

log("Importing local modules...")
from dataloader import load_split_data, ReadmissionDataset
from lstm_model import create_lstm_model
log("✓ Local modules imported")

log("Setting up paths...")
BASE_DIR = Path(__file__).parent.parent / 'data'
log(f"BASE_DIR: {BASE_DIR}")

log("Loading data...")
start = time.time()
train_ids, static_df, tensor_dir = load_split_data('train', BASE_DIR)
log(f"✓ Loaded train in {time.time()-start:.1f}s: {len(train_ids)} IDs")

log("Creating dataset...")
start = time.time()
train_dataset = ReadmissionDataset(train_ids, static_df, tensor_dir, use_time_series=True)
log(f"✓ Created dataset in {time.time()-start:.1f}s")

log("Creating DataLoader...")
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
log(f"✓ Created DataLoader with {len(train_loader)} batches")

log("Creating LSTM model...")
start = time.time()
model = create_lstm_model()
log(f"✓ Created model in {time.time()-start:.1f}s")
log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

log("Testing first batch...")
start = time.time()
batch = next(iter(train_loader))
log(f"✓ Loaded first batch in {time.time()-start:.1f}s")
log(f"  X shape: {batch['X'].shape}")
log(f"  mask shape: {batch['mask'].shape}")
log(f"  static shape: {batch['static'].shape}")

log("Testing forward pass...")
start = time.time()
X = batch['X']
mask = batch['mask']
static = batch['static']
output = model(X, mask, static)
log(f"✓ Forward pass in {time.time()-start:.1f}s")
log(f"  Output shape: {output.shape}")

log("")
log("="*70)
log("ALL TESTS PASSED!")
log("="*70)
log_file.close()
