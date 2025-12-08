import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys
sys.path.append('src/analysis')
from rolling_correlation import load_asset_data, calculate_rolling_correlation

# Check if tensorflow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not installed. Install with: pip install tensorflow")

class LSTMCorrelationPredictor:
    """
    LSTM-based correlation predictor
    Uses sequence modeling to capture temporal dependencies
    """
    
    def __init__(self, lookback=30, lstm_units=64):
        """
        Initialize LSTM predictor
        
        Args:
            lookback: Number of past days to use as features
            lstm_units: Number of LSTM units
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM predictor")
        
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def build_model(self, input_shape):
        """Build LSTM architecture"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1)  # Output: predicted correlation
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, correlation_series):
        """
        Prepare sequences for LSTM
        
        Args:
            correlation_series: Time series of correlations
        
        Returns:
            X: Sequences (samples, lookback, features)
            y: Target values
        """
        # Remove NaN values first
        clean_series = correlation_series.dropna()
        
        if len(clean_series) < self.lookback + 1:
            raise ValueError(f"Not enough data after removing NaN values. Need at least {self.lookback + 1}, got {len(clean_series)}")
        
        X, y = [], []
        
        for i in range(self.lookback, len(clean_series)):
            # Features: past 'lookback' days
            X.append(clean_series.iloc[i-self.lookback:i].values)
            # Target: next day
            y.append(clean_series.iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM: (samples, lookback, 1)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train(self, correlation_series, test_size=0.2, epochs=50, batch_size=32, verbose=0):
        """
        Train LSTM model
        
        Args:
            correlation_series: Historical correlation time series
            test_size: Fraction for testing
            epochs: Training epochs
            batch_size: Batch size
            verbose: Verbosity level
        
        Returns:
            Dictionary with training metrics
        """
        # Prepare sequences (this now handles NaN removal internally)
        X, y = self.prepare_sequences(correlation_series)
        
        if len(X) == 0:
            raise ValueError("Not enough data to train model")
        
        # Train/test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        
        # Build model
        self.model = self.build_model(input_shape=(self.lookback, 1))
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled, verbose=0).flatten()
        test_pred = self.model.predict(X_test_scaled, verbose=0).flatten()
        
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': np.mean(np.abs(y_train - train_pred)),
            'test_mae': np.mean(np.abs(y_test - test_pred)),
            'lookback': self.lookback,
            'epochs_trained': len(history.history['loss'])
        }
        
        return metrics
    
    def predict_future(self, correlation_series, days_ahead=30):
        """
        Predict future correlations
        
        Args:
            correlation_series: Recent correlation history
            days_ahead: How many days to predict
        
        Returns:
            Array of predicted correlations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Remove NaN and use the most recent data
        clean_series = correlation_series.dropna()
        recent_data = clean_series.iloc[-self.lookback:].values.copy()
        predictions = []
        
        # Iterative prediction
        for _ in range(days_ahead):
            # Prepare sequence
            X = recent_data.reshape(1, self.lookback, 1)
            X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
            
            # Predict next value
            next_pred = self.model.predict(X_scaled, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence
            recent_data = np.append(recent_data[1:], next_pred)
        
        return np.array(predictions)
    
    def get_prediction_confidence(self, correlation_series):
        """Calculate prediction confidence based on recent errors"""
        if not self.is_trained:
            return None
        
        # Remove NaN values
        clean_series = correlation_series.dropna()
        
        # Use last 100 points (or all if less than 100)
        test_data = clean_series.iloc[-min(100, len(clean_series)):]
        X, y = self.prepare_sequences(test_data)
        
        if len(X) == 0:
            return None
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        errors = np.abs(y - predictions)
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }

def predict_correlation_lstm(asset1_data, asset2_data, window=90, days_ahead=30, 
                             lookback=30, epochs=50):
    """
    Convenience function to train and predict with LSTM
    
    Args:
        asset1_data: Price series for asset 1
        asset2_data: Price series for asset 2
        window: Rolling window for correlation
        days_ahead: Prediction horizon
        lookback: LSTM lookback period
        epochs: Training epochs
    
    Returns:
        Dictionary with predictions and metrics
    """
    # Calculate rolling correlation
    rolling_corr = calculate_rolling_correlation(asset1_data, asset2_data, window=window)
    
    # Check if we have enough data after removing NaN
    clean_corr = rolling_corr.dropna()
    if len(clean_corr) < 200:
        raise ValueError(f"Need at least 200 days of clean data for LSTM, got {len(clean_corr)}")
    
    # Initialize and train
    predictor = LSTMCorrelationPredictor(lookback=lookback)
    print(f"Training LSTM model (this may take 1-2 minutes)...")
    metrics = predictor.train(rolling_corr, epochs=epochs, verbose=0)
    
    # Make predictions
    predictions = predictor.predict_future(rolling_corr, days_ahead=days_ahead)
    confidence = predictor.get_prediction_confidence(rolling_corr)
    
    # Create date index
    last_date = rolling_corr.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
    
    return {
        'predictions': predictions,
        'dates': future_dates,
        'current_correlation': rolling_corr.dropna().iloc[-1],
        'metrics': metrics,
        'confidence': confidence,
        'model_type': 'lstm'
    }

if __name__ == "__main__":
    if not TENSORFLOW_AVAILABLE:
        print("Please install tensorflow: pip install tensorflow")
        exit(1)
    
    print("Testing LSTM Predictor...\n")
    
    # Load sample data
    spy = load_asset_data('data/raw/SPY_etf.csv')
    gold = load_asset_data('data/raw/GCF_commodity.csv')
    
    # Make prediction
    result = predict_correlation_lstm(spy, gold, window=90, days_ahead=30, lookback=30, epochs=30)
    
    print(f"\nModel: {result['model_type'].upper()}")
    print(f"Current Correlation: {result['current_correlation']:.3f}")
    print(f"Predicted Correlation (30 days): {result['predictions'][-1]:.3f}")
    print(f"\nModel Performance:")
    print(f"  Train R²: {result['metrics']['train_r2']:.3f}")
    print(f"  Test R²: {result['metrics']['test_r2']:.3f}")
    print(f"  Test MAE: {result['metrics']['test_mae']:.3f}")
    print(f"  Epochs trained: {result['metrics']['epochs_trained']}")
    print(f"\nPrediction Confidence:")
    print(f"  Mean Error: ±{result['confidence']['mean_error']:.3f}")