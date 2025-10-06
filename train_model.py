"""
Main training script - Run this to train the model
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
import json

def main():
    print("="*60)
    print("STOCK MARKET PREDICTOR - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Download data
    print("\n[1/5] Downloading data...")
    loader = StockDataLoader(ticker="SPY")
    raw_data = loader.download_data(start_date="2015-01-01")
    
    # Step 2: Feature engineering
    print("\n[2/5] Engineering features...")
    engineer = FeatureEngineer(raw_data)
    features_df = engineer.create_all_features()
    print(f"Created {features_df.shape[1]} features")
    
    # Step 3: Load config
    print("\n[3/5] Loading configuration...")
    with open("configs/training_config.json", 'r') as f:
        config = json.load(f)
    
    # Step 4: Initialize trainer
    print("\n[4/5] Initializing trainer...")
    trainer = ModelTrainer(config)
    
    # Step 5: Prepare data
    print("\n[5/5] Training model...")
    feature_cols = engineer.get_feature_names()
    X = features_df[feature_cols].values
    y = features_df['target'].values
    
    # Normalize
    X_scaled = trainer.scaler.fit_transform(X)
    
    # Create sequences
    X_seq, y_seq = trainer.prepare_sequences(X_scaled, y)
    
    # Split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train
    final_model = trainer.train_final_model(X_train, y_train, X_val, y_val)
    
    # Save
    trainer.save_scaler("artifacts/scaler.pkl")
    
    # Save feature names
    with open("artifacts/feature_names.json", 'w') as f:
        json.dump(feature_cols, f)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: models/")
    print(f"Scaler saved to: artifacts/scaler.pkl")
    print(f"Ready for deployment!")


if __name__ == "__main__":
    main()