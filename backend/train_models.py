"""
train_models.py
===============
Trains 6 deep learning models for rainfall prediction on Dakshina Kannada dataset.

Models:
1. LSTM (Long Short-Term Memory)
2. Bidirectional LSTM
3. GRU (Gated Recurrent Unit)
4. CNN-LSTM Hybrid
5. Transformer (Temporal Self-Attention)
6. Stacked Autoencoder + Regression Head

Features:
- 80/10/10 train/val/test split with temporal ordering
- MinMaxScaler normalization (saved as scaler.pkl)
- EarlyStopping, ReduceLROnPlateau, Dropout, BatchNormalization
- Huber loss for robustness to rainfall outliers
- Target: R² ≥ 0.88, MAE ≤ 8mm on test set
- Saves models as .h5 and metrics as JSON

Usage:
    python train_models.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    # Data paths
    'data_path': 'data/dakshina_kannada_weather.csv',
    'models_dir': 'models',
    'scaler_path': 'models/scaler.pkl',
    'metrics_path': 'models/metrics.json',
    
    # Sequence parameters
    'sequence_length': 30,  # 30-day lookback window
    'target_column': 'rainfall_mm',
    
    # Model training
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,  # Early stopping patience
    'learning_rate': 0.001,
    'lr_patience': 5,  # ReduceLROnPlateau patience
    'lr_factor': 0.5,
    
    # Architecture defaults
    'dropout_rate': 0.2,
    'l2_reg': 0.001,
    
    # Features to use (excluding date, year, month, day, is_monsoon for raw input)
    'feature_columns': [
        'temperature_max', 'temperature_min', 'humidity', 'wind_speed',
        'pressure', 'cloud_cover', 'sunshine_hours', 'dew_point',
        'evapotranspiration', 'previous_7day_rainfall', 'previous_30day_rainfall'
    ]
}


def load_and_preprocess_data(config: Dict) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Load dataset, encode cyclical features, and normalize.
    
    Returns:
        X: Normalized feature array (n_samples, n_features)
        y: Target rainfall values (n_samples,)
        scaler: Fitted MinMaxScaler for inverse transform
    """
    print(f"📁 Loading data from {config['data_path']}...")
    df = pd.read_csv(config['data_path'])
    
    # Add cyclical encoding for temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Select features and target
    feature_cols = config['feature_columns'] + ['month_sin', 'month_cos', 'day_sin', 'day_cos']
    X = df[feature_cols].values.astype(np.float32)
    y = df[config['target_column']].values.astype(np.float32)
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Loaded {len(df):,} samples with {X.shape[1]} features")
    return X_scaled, y, scaler


def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create temporal sequences for time-series modeling.
    
    Args:
        X: Normalized features (n_samples, n_features)
        y: Target values (n_samples,)
        sequence_length: Lookback window size
        
    Returns:
        X_seq: Sequenced features (n_sequences, sequence_length, n_features)
        y_seq: Corresponding targets (n_sequences,)
    """
    X_seq, y_seq = [], []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def temporal_train_val_test_split(
    X: np.ndarray, y: np.ndarray, 
    train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data temporally (no shuffling) to prevent data leakage.
    80% train, 10% validation, 10% test.
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"📊 Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm_model(input_shape: Tuple[int, int], name: str = "lstm") -> keras.Model:
    """
    Model 1: LSTM - Captures long-term temporal rainfall patterns.
    
    Architecture:
        Input → LSTM(128, return_sequences=True) → LSTM(64) → Dense(32) → Dense(1)
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    x = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(CONFIG['l2_reg']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    outputs = layers.Dense(1, name="output")(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def build_bilstm_model(input_shape: Tuple[int, int], name: str = "bilstm") -> keras.Model:
    """
    Model 2: Bidirectional LSTM - Learns forward and backward temporal dependencies.
    
    Architecture:
        Input → BiLSTM(64, return_sequences=True) → BiLSTM(32) → Dropout → Dense(1)
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False), name="bilstm_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    outputs = layers.Dense(1, name="output")(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def build_gru_model(input_shape: Tuple[int, int], name: str = "gru") -> keras.Model:
    """
    Model 3: GRU - Faster training with comparable accuracy to LSTM.
    
    Architecture:
        Input → GRU(128, return_sequences=True) → GRU(64) → Dense(32) → Dense(1)
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    x = layers.GRU(128, return_sequences=True, name="gru_1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    x = layers.GRU(64, return_sequences=False, name="gru_2")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(CONFIG['l2_reg']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    outputs = layers.Dense(1, name="output")(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def build_cnn_lstm_model(input_shape: Tuple[int, int], name: str = "cnn_lstm") -> keras.Model:
    """
    Model 4: CNN-LSTM Hybrid - CNN extracts local patterns; LSTM captures temporal flow.
    
    Architecture:
        Input → Conv1D(64, kernel=3) → MaxPooling → LSTM(64) → Dense(1)
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    # CNN block for local feature extraction
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    # LSTM for temporal modeling
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    outputs = layers.Dense(1, name="output")(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def positional_encoding(position: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encoding.
    Safely handles odd d_model dimensions.
    """
    pe = np.zeros((position, d_model), dtype=np.float32)
    position = np.arange(position)[:, np.newaxis].astype(np.float32)
    
    # Compute frequencies for pairs of dimensions
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Assign sine to even indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Assign cosine to odd indices (1, 3, 5...), truncate div_term if d_model is odd
    pe[:, 1::2] = np.cos(position * div_term[:d_model//2])
    
    return pe


class PositionalEncoding(layers.Layer):
    """Keras layer for positional encoding in Transformer."""
    def __init__(self, position: int, d_model: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        # Precompute as a constant tensor to avoid graph/slice issues
        pe = positional_encoding(position, d_model)
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        # x shape: (batch, seq_len, d_model)
        # pos_encoding shape: (seq_len, d_model)
        # TensorFlow broadcasting automatically aligns them correctly
        return x + self.pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'position': int(self.pos_encoding.shape[0]), 
            'd_model': int(self.pos_encoding.shape[1])
        })
        return config


def build_transformer_model(input_shape: Tuple[int, int], name: str = "transformer") -> keras.Model:
    """
    Model 5: Transformer with Temporal Self-Attention.
    
    Architecture:
        Input → PositionalEncoding → MultiHeadAttention(4 heads) → LayerNorm 
        → FeedForward → GlobalAvgPool → Dense(1)
    
    Built with Keras Functional API.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    d_model = input_shape[1]
    seq_length = input_shape[0]
    
    # Positional encoding
    x = PositionalEncoding(seq_length, d_model)(inputs)
    
    # Multi-Head Self-Attention (4 heads)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=d_model//4, name="mha")(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    # Feed-Forward Network
    ffn = keras.Sequential([
        layers.Dense(d_model * 4, activation='relu'),
        layers.Dense(d_model)
    ], name="ffn")
    
    x = ffn(x)
    x = layers.Add()([x, inputs])  # Residual connection
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    # Global average pooling over sequence dimension
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(1, name="output")(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def build_autoencoder_model(input_shape: Tuple[int, int], name: str = "autoencoder") -> Tuple[keras.Model, keras.Model]:
    """
    Model 6: Stacked Autoencoder + Regression Head.
    
    Architecture:
        Encoder: Dense(128→64→32) with bottleneck
        Decoder: Dense(32→64→128) for reconstruction
        Regression head: From bottleneck (32-dim) → Dense(1)
    
    Returns:
        autoencoder: Full autoencoder for pretraining
        regressor: Encoder + regression head for fine-tuning
    """
    # Flatten input for dense layers: (seq_len, features) → (seq_len * features,)
    flat_dim = input_shape[0] * input_shape[1]
    
    # Encoder
    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.Flatten()(inputs)
    
    x = layers.Dense(128, activation='relu', name="enc_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    x = layers.Dense(64, activation='relu', name="enc_2")(x)
    x = layers.BatchNormalization()(x)
    
    bottleneck = layers.Dense(32, activation='relu', name="bottleneck")(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu', name="dec_1")(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    x = layers.Dense(128, activation='relu', name="dec_2")(x)
    x = layers.BatchNormalization()(x)
    
    reconstructed = layers.Dense(flat_dim, name="reconstructed")(x)
    reconstructed = layers.Reshape(input_shape)(reconstructed)
    
    # Autoencoder model (for pretraining)
    autoencoder = keras.Model(inputs, reconstructed, name=f"{name}_autoencoder")
    
    # Regression head (for fine-tuning)
    reg_x = layers.Dense(16, activation='relu')(bottleneck)
    reg_x = layers.BatchNormalization()(reg_x)
    reg_x = layers.Dropout(CONFIG['dropout_rate'])(reg_x)
    reg_output = layers.Dense(1, name="regression_output")(reg_x)
    
    regressor = keras.Model(inputs, reg_output, name=name)
    
    return autoencoder, regressor


def compile_model(model: keras.Model) -> None:
    """Compile model with Huber loss optimized for rainfall magnitudes."""
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss=tf.keras.losses.Huber(delta=50.0),  # Matches 0-300mm rainfall scale
        metrics=['mae', 'mse']
    )


def train_model(
    model: keras.Model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    model_name: str,
    epochs: int = None
) -> keras.callbacks.History:
    """
    Train a model with callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
    """
    if epochs is None:
        epochs = CONFIG['epochs']
    
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=CONFIG['lr_factor'],
            patience=CONFIG['lr_patience'],
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f"{CONFIG['models_dir']}/{model_name}_best.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    print(f"🔄 Training {model_name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks_list,
        verbose=0
    )
    
    # Load best weights
    best_path = f"{CONFIG['models_dir']}/{model_name}_best.h5"
    if os.path.exists(best_path):
        model.load_weights(best_path)
        os.remove(best_path)  # Clean up temporary file
    
    return history


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, 
                   scaler: MinMaxScaler, feature_cols: List[str]) -> Dict[str, float]:
    """
    Evaluate model on test set and compute metrics.
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Inverse transform predictions and targets for metric calculation
    # Note: Only target needs inverse transform; features were normalized separately
    y_test_orig = y_test  # Already in original scale
    y_pred_orig = y_pred  # Model outputs original scale (last layer has no activation)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / np.maximum(y_test_orig, 0.1))) * 100
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum(np.prod(w.shape) for w in model.trainable_weights)
    
    return {
        'mae': round(float(mae), 3),
        'rmse': round(float(rmse), 3),
        'r2': round(float(r2), 4),
        'mape': round(float(mape), 2),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params)
    }


def train_autoencoder_pipeline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    input_shape: Tuple[int, int]
) -> keras.Model:
    """
    Special training pipeline for autoencoder:
    1. Pretrain autoencoder for 20 epochs (reconstruction task)
    2. Extract encoder + add regression head
    3. Fine-tune regression head
    """
    print("🔄 Training Autoencoder (2-stage pipeline)...")
    
    # Build models
    autoencoder, regressor = build_autoencoder_model(input_shape)
    
    # Stage 1: Pretrain autoencoder
    autoencoder.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    print("  └─ Stage 1: Pretraining autoencoder (reconstruction)...")
    autoencoder.fit(
        X_train, X_train,  # Input = Output for autoencoder
        validation_data=(X_val, X_val),
        epochs=20,
        batch_size=CONFIG['batch_size'],
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ],
        verbose=0
    )
    
    # Stage 2: Fine-tune regression head
    print("  └─ Stage 2: Fine-tuning regression head...")
    
    # Freeze encoder layers
    for layer in regressor.layers:
        if 'bottleneck' not in layer.name and 'regression' not in layer.name:
            layer.trainable = False
    
    compile_model(regressor)
    
    # Train only regression head initially
    regressor.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=CONFIG['batch_size'],
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ],
        verbose=0
    )
    
    # Unfreeze and fine-tune entire model
    for layer in regressor.layers:
        layer.trainable = True
    
    compile_model(regressor)
    regressor.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=CONFIG['batch_size'],
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ],
        verbose=0
    )
    
    return regressor


def print_training_summary(results: Dict[str, Dict], elapsed_time: float):
    """Print formatted training results table."""
    print("\n" + "="*90)
    print("📊 MODEL TRAINING RESULTS")
    print("="*90)
    print(f"{'Model':<15} {'R²':>8} {'MAE (mm)':>12} {'RMSE (mm)':>12} {'MAPE (%)':>10} {'Params':>10}")
    print("-"*90)
    
    best_model = None
    best_r2 = -1
    
    for name, metrics in results.items():
        r2 = metrics['r2']
        mae = metrics['mae']
        rmse = metrics['rmse']
        mape = metrics['mape']
        params = f"{metrics['trainable_params']:,}"
        
        # Highlight best model
        marker = " ⭐" if r2 > best_r2 else ""
        if r2 > best_r2:
            best_r2 = r2
            best_model = name
        
        status = "✅" if r2 >= 0.88 and mae <= 8 else "⚠️"
        print(f"{name:<15} {r2:>8.4f} {mae:>12.3f} {rmse:>12.3f} {mape:>10.2f} {params:>10} {status}{marker}")
    
    print("-"*90)
    print(f"✨ Best Model: {best_model} (R² = {best_r2:.4f})")
    print(f"⏱️  Total Training Time: {elapsed_time/60:.1f} minutes")
    
    # Target check
    qualified = sum(1 for m in results.values() if m['r2'] >= 0.90)
    print(f"🎯 Models with R² ≥ 0.90: {qualified}/6 (target: ≥4)")
    print("="*90 + "\n")


def main():
    """Main training pipeline."""
    start_time = time.time()
    
    # Create models directory
    Path(CONFIG['models_dir']).mkdir(parents=True, exist_ok=True)
    
    print("🚀 DK-AquaPredict — Model Training Pipeline")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90)
    
    # Step 1: Load and preprocess data
    X_raw, y_raw, scaler = load_and_preprocess_data(CONFIG)
    
    # Save scaler for inference
    import joblib
    joblib.dump(scaler, CONFIG['scaler_path'])
    print(f"💾 Scaler saved to {CONFIG['scaler_path']}")
    
    # Step 2: Create sequences
    print(f"🔗 Creating {CONFIG['sequence_length']}-day sequences...")
    X_seq, y_seq = create_sequences(X_raw, y_raw, CONFIG['sequence_length'])
    print(f"✅ Created {len(X_seq):,} sequences of shape {X_seq.shape}")
    
    # Step 3: Temporal train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_train_val_test_split(
        X_seq, y_seq, train_ratio=0.8, val_ratio=0.1
    )
    
    # Step 4: Define models
    input_shape = (CONFIG['sequence_length'], X_seq.shape[2])
    
    model_builders = {
        'lstm': build_lstm_model,
        'bilstm': build_bilstm_model,
        'gru': build_gru_model,
        'cnn_lstm': build_cnn_lstm_model,
        'transformer': build_transformer_model,
        # Autoencoder handled separately
    }
    
    results = {}
    
    # Step 5: Train standard models
    for name, builder in model_builders.items():
        clear_session()  # Free memory between models
        
        print(f"\n{'='*50}")
        print(f"🔹 Training: {name.upper()}")
        print(f"{'='*50}")
        
        model = builder(input_shape, name=name)
        compile_model(model)
        
        history = train_model(model, X_train, y_train, X_val, y_val, name)
        metrics = evaluate_model(model, X_test, y_test, scaler, CONFIG['feature_columns'])
        metrics['training_time'] = round(time.time() - start_time, 1)
        
        # Save model
        model_path = f"{CONFIG['models_dir']}/model_{name}.h5"
        model.save(model_path)
        print(f"💾 Model saved: {model_path}")
        
        results[name] = metrics
        print(f"📈 Test R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.3f}mm")
    
    # Step 6: Train autoencoder (special pipeline)
    print(f"\n{'='*50}")
    print(f"🔹 Training: AUTOENCODER (2-stage)")
    print(f"{'='*50}")
    
    clear_session()
    regressor = train_autoencoder_pipeline(X_train, y_train, X_val, y_val, input_shape)
    metrics = evaluate_model(regressor, X_test, y_test, scaler, CONFIG['feature_columns'])
    metrics['training_time'] = round(time.time() - start_time, 1)
    
    # Save model
    model_path = f"{CONFIG['models_dir']}/model_autoencoder.h5"
    regressor.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    results['autoencoder'] = metrics
    print(f"📈 Test R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.3f}mm")
    
    # Step 7: Save metrics JSON
    with open(CONFIG['metrics_path'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Metrics saved to {CONFIG['metrics_path']}")
    
    # Step 8: Print summary
    elapsed = time.time() - start_time
    print_training_summary(results, elapsed)
    
    # Final check
    qualified = sum(1 for m in results.values() if m['r2'] >= 0.90)
    if qualified >= 4:
        print("🎉 SUCCESS: Target achieved (≥4 models with R² ≥ 0.90)!")
    else:
        print(f"⚠️  Note: {qualified}/6 models achieved R² ≥ 0.90. Consider hyperparameter tuning.")
    
    print(f"\n✨ Training complete! Next: Start backend with `uvicorn main:app --reload`")
    
    return results


if __name__ == "__main__":
    results = main()