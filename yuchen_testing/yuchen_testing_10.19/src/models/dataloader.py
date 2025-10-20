"""
DataLoader for MIMIC-IV Readmission Prediction

Handles loading time-series (NPZ) and static features (CSV) for training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch


class ReadmissionDataset(Dataset):
    """
    Dataset for 30-day readmission prediction.
    
    Loads:
    - Time-series lab data from NPZ files (48 x 11 with masks)
    - Static features from CSV
    - Labels
    """
    
    def __init__(self, hadm_ids, static_df, tensor_dir, use_time_series=True):
        """
        Args:
            hadm_ids: List of admission IDs for this split
            static_df: DataFrame with static features (must contain hadm_id)
            tensor_dir: Directory containing NPZ files
            use_time_series: Whether to load time-series data (False for static-only models)
        """
        self.hadm_ids = hadm_ids
        self.use_time_series = use_time_series
        self.tensor_dir = Path(tensor_dir)
        
        # Filter static features to this split
        self.static_df = static_df[static_df['hadm_id'].isin(hadm_ids)].copy()
        self.static_df = self.static_df.set_index('hadm_id')
        
        # Convert categorical admission_type to numeric
        admission_type_map = {'emergency': 0, 'elective': 1, 'observation': 2, 'other': 3}
        self.static_df['admission_type_cat'] = self.static_df['admission_type_cat'].map(admission_type_map)
        
        # Define static feature columns
        self.static_cols = ['age', 'gender_binary', 'los_days', 
                           'admission_type_cat', 'charlson_score', 'num_diagnoses']
        
        print(f"Dataset initialized: {len(self.hadm_ids)} admissions")
        print(f"  Use time-series: {self.use_time_series}")
        print(f"  Static features: {self.static_cols}")
        
    def __len__(self):
        return len(self.hadm_ids)
    
    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        
        # Load static features
        static_row = self.static_df.loc[hadm_id]
        static_features = static_row[self.static_cols].values.astype(np.float32)
        label = static_row['readmit_30d']
        
        if self.use_time_series:
            # Load time-series from NPZ
            # Try multiple filename formats (some have .0 suffix, some don't)
            npz_file = self.tensor_dir / f"hadm_{hadm_id}.npz"
            if not npz_file.exists():
                npz_file = self.tensor_dir / f"hadm_{hadm_id}.0.npz"
            
            if npz_file.exists():
                data = np.load(npz_file)
                X = data['X'].astype(np.float32)  # (48, 11)
                mask = data['mask'].astype(np.float32)  # (48, 11)
                
                # Replace NaN with 0 (NaN represents missing values)
                X = np.nan_to_num(X, nan=0.0)
            else:
                # Missing file - create zero tensor with all masked
                X = np.zeros((48, 11), dtype=np.float32)
                mask = np.zeros((48, 11), dtype=np.float32)
            
            return {
                'X': torch.from_numpy(X),
                'mask': torch.from_numpy(mask),
                'static': torch.from_numpy(static_features),
                'label': torch.tensor(label, dtype=torch.float32),
                'hadm_id': hadm_id
            }
        else:
            # Static features only
            return {
                'static': torch.from_numpy(static_features),
                'label': torch.tensor(label, dtype=torch.float32),
                'hadm_id': hadm_id
            }


def load_split_data(split_name, base_dir):
    """
    Load data for a specific split (train/val/test).
    
    Args:
        split_name: 'train', 'val', or 'test'
        base_dir: Base directory containing processed_data/
        
    Returns:
        hadm_ids: List of admission IDs
        static_df: DataFrame with all static features
        tensor_dir: Path to time-series tensors
    """
    base_dir = Path(base_dir)
    processed_dir = base_dir / 'mimic_data' / 'processed_data'
    
    # Load hadm_ids for this split
    split_file = processed_dir / f'split_{split_name}.txt'
    hadm_ids = pd.read_csv(split_file, header=None)[0].astype(int).tolist()
    
    # Load static features
    static_file = processed_dir / 'static_features.csv'
    static_df = pd.read_csv(static_file)
    
    # Tensor directory
    tensor_dir = processed_dir / 'time_series_tensors'
    
    return hadm_ids, static_df, tensor_dir


def create_dataloaders(base_dir, batch_size=64, use_time_series=True, num_workers=0):
    """
    Create train/val/test dataloaders.
    
    Args:
        base_dir: Base directory containing src/data/
        batch_size: Batch size for training
        use_time_series: Whether to load time-series data
        num_workers: Number of workers for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # Load data for each split
    train_ids, static_df, tensor_dir = load_split_data('train', base_dir)
    val_ids, _, _ = load_split_data('val', base_dir)
    test_ids, _, _ = load_split_data('test', base_dir)
    
    # Create datasets
    train_dataset = ReadmissionDataset(train_ids, static_df, tensor_dir, use_time_series)
    val_dataset = ReadmissionDataset(val_ids, static_df, tensor_dir, use_time_series)
    test_dataset = ReadmissionDataset(test_ids, static_df, tensor_dir, use_time_series)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataloader
    print("Testing DataLoader...")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent / 'data'
    
    # Test static-only dataset
    print("\n1. Testing STATIC-ONLY dataset:")
    print("-"*60)
    train_ids, static_df, tensor_dir = load_split_data('train', base_dir)
    dataset_static = ReadmissionDataset(train_ids[:1000], static_df, tensor_dir, use_time_series=False)
    
    sample = dataset_static[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Static features shape: {sample['static'].shape}")
    print(f"Static features: {sample['static']}")
    print(f"Label: {sample['label'].item()}")
    
    # Test time-series dataset
    print("\n2. Testing TIME-SERIES dataset:")
    print("-"*60)
    dataset_ts = ReadmissionDataset(train_ids[:1000], static_df, tensor_dir, use_time_series=True)
    
    sample = dataset_ts[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"X shape: {sample['X'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Static features shape: {sample['static'].shape}")
    print(f"Non-zero mask values: {sample['mask'].sum().item()}")
    
    # Test dataloader batching
    print("\n3. Testing DATALOADER batching:")
    print("-"*60)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset_ts, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch X shape: {batch['X'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
    print(f"Batch static shape: {batch['static'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    print(f"Label distribution: {batch['label'].mean().item():.3f}")
    
    print("\n" + "="*60)
    print("✓ DataLoader tests passed!")
    print("="*60)
