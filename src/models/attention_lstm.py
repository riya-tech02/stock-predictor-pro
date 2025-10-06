"""
Advanced Bi-LSTM with Attention Mechanism for Stock Price Prediction
Implements state-of-the-art sequence modeling with interpretable attention weights
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class AttentionLayer(layers.Layer):
    """
    Custom attention mechanism to focus on important time steps
    Returns attention-weighted context vector and attention weights for explainability
    """
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, activation='tanh')
        self.V = layers.Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        score = self.V(self.W(inputs))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class StockPredictorModel:
    """Production-grade stock prediction model with Bi-LSTM + Attention"""
    
    def __init__(self, sequence_length=60, n_features=10, lstm_units=[128, 64],
                 attention_units=128, dropout_rate=0.3, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.attention_model = None
        
    def build_model(self):
        """Build the model architecture"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features), name='sequence_input')
        
        x = layers.Bidirectional(layers.LSTM(self.lstm_units[0], return_sequences=True,
                                             kernel_regularizer=keras.regularizers.l2(0.001)), name='bilstm_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Bidirectional(layers.LSTM(self.lstm_units[1], return_sequences=True,
                                             kernel_regularizer=keras.regularizers.l2(0.001)), name='bilstm_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        context_vector, attention_weights = AttentionLayer(units=self.attention_units, name='attention')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_1')(context_vector)
        x = layers.Dropout(self.dropout_rate, name='dropout_3')(x)
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        
        outputs = layers.Dense(3, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='stock_predictor')
        self.attention_model = Model(inputs=inputs, outputs=attention_weights, name='attention_weights')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def get_attention_weights(self, X):
        """Extract attention weights for explainability"""
        if self.attention_model is None:
            raise ValueError("Model must be built before extracting attention")
        return self.attention_model.predict(X, verbose=0)
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath, custom_objects={'AttentionLayer': AttentionLayer})
        attention_layer = self.model.get_layer('attention')
        self.attention_model = Model(inputs=self.model.input, outputs=attention_layer.output[1])
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model architecture"""
        return self.model.summary()