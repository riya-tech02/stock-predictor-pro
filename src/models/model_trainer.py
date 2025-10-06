"""
Simplified Model Training Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from typing import Tuple
import os
import json

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.attention_lstm import StockPredictorModel


class ModelTrainer:
    """Training pipeline"""
    
    def __init__(self, config: dict):
        self.config = config
        self.sequence_length = config.get('sequence_length', 60)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 50)
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(targets[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train_final_model(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        
        print(f"\nTraining Shape: {X_train.shape}")
        print(f"Validation Shape: {X_val.shape}")
        
        # Convert targets to categorical
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=3)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=3)
        
        # Build model
        self.model = StockPredictorModel(
            sequence_length=self.sequence_length,
            n_features=X_train.shape[2],
            lstm_units=self.config.get('lstm_units', [128, 64]),
            attention_units=self.config.get('attention_units', 128),
            dropout_rate=self.config.get('dropout_rate', 0.3),
            learning_rate=self.config.get('learning_rate', 0.001)
        )
        self.model.build_model()
        
        print("\n" + "="*60)
        print("Training Model...")
        print("="*60)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_metrics = self.model.model.evaluate(X_val, y_val_cat, verbose=0)
        
        print(f"\n✓ Training Complete!")
        print(f"  Validation Accuracy: {val_metrics[1]:.4f}")
        print(f"  Validation Loss: {val_metrics[0]:.4f}")
        
        return self.model
    
    def save_scaler(self, filepath: str = "artifacts/scaler.pkl"):
        """Save fitted scaler"""
        import joblib
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        print(f"✓ Scaler saved to {filepath}")