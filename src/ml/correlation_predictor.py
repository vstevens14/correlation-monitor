import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys
sys.path.append('../analysis')
from rolling_correlation import load_asset_data, calculate_rolling_correlation

class CorrelationPredictor:
    """
    Modular ML framework for predicting future correlations
    Supports multiple models that can be easily swapped/compared
    """
    
    def __init__(self, model_type='linear'):
        """
        Initialize predictor with specified model type
        
        Args:
            model_type: 'linear', 'lstm', 'random_forest' (more to be added)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize the selected model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'lstm':
            raise NotImplementedError("LSTM model coming soon!")
        elif model_type == 'random_forest':
            raise NotImplementedError("Random Forest model coming soon!")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, correlation_series, lookback=30):
        """
        Prepare features for ML model
        Uses sliding window approach: past N days predict next day
        
        Args:
            correlation_series: Time series of correlations
            lookback: How many past days to use as features
        
        Returns:
            X: Features (past correlations)
            y: Target (next day correlation)
        """
        # Remove NaN values first
        correlation_series = correlation_series.dropna()
        
        X, y = [], []
        
        for i in range(lookback, len(correlation_series)):
            # Features: past 'lookback' days of correlations
            window = correlation_series.iloc[i-lookback:i].values
            target = correlation_series.iloc[i]
            
            # Only add if no NaN values in window or target
            if not np.isnan(window).any() and not np.isnan(target):
                X.append(window)
                y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self, correlation_series, lookback=30, test_size=0.2):
        """
        Train the model on historical correlation data
        
        Args:
            correlation_series: Historical correlation time series
            lookback: Number of past days to use as features
            test_size: Fraction of data to use for testing
        
        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        X, y = self.prepare_features(correlation_series, lookback=lookback)
        
        if len(X) == 0:
            raise ValueError("Not enough data to train model")
        
        # Train/test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'lookback': lookback
        }
        
        return metrics
    
    def predict_future(self, correlation_series, days_ahead=30, lookback=30):
        """
        Predict future correlations
        
        Args:
            correlation_series: Recent correlation history
            days_ahead: How many days into the future to predict
            lookback: Number of past days to use (must match training)
        
        Returns:
            Array of predicted correlations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Use the most recent data as starting point
        recent_data = correlation_series.iloc[-lookback:].values.copy()
        predictions = []
        
        # Iterative prediction: predict next day, then use that prediction for the next
        for _ in range(days_ahead):
            # Prepare features
            X = recent_data.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predict next value
            next_pred = self.model.predict(X_scaled)[0]
            predictions.append(next_pred)
            
            # Update recent_data: remove oldest, add newest prediction
            recent_data = np.append(recent_data[1:], next_pred)
        
        return np.array(predictions)
    
    def get_prediction_confidence(self, correlation_series, lookback=30):
        """
        Calculate confidence interval for predictions
        Returns standard deviation of recent prediction errors
        """
        if not self.is_trained:
            return None
        
        # Use last 100 points to estimate prediction error
        X, y = self.prepare_features(correlation_series.iloc[-100:], lookback=lookback)
        if len(X) == 0:
            return None
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        errors = np.abs(y - predictions)
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }

def predict_correlation(asset1_data, asset2_data, window=90, days_ahead=30, model_type='linear'):
    """
    Convenience function to train and predict correlation
    
    Args:
        asset1_data: Price series for asset 1
        asset2_data: Price series for asset 2
        window: Rolling window for correlation calculation
        days_ahead: How many days to predict
        model_type: Type of ML model to use
    
    Returns:
        Dictionary with predictions and metrics
    """
    # Calculate historical rolling correlation
    rolling_corr = calculate_rolling_correlation(asset1_data, asset2_data, window=window)
    
    if len(rolling_corr) < 100:
        raise ValueError("Need at least 100 days of data for prediction")
    
    # Initialize and train predictor
    predictor = CorrelationPredictor(model_type=model_type)
    metrics = predictor.train(rolling_corr, lookback=30, test_size=0.2)
    
    # Make predictions
    predictions = predictor.predict_future(rolling_corr, days_ahead=days_ahead, lookback=30)
    confidence = predictor.get_prediction_confidence(rolling_corr, lookback=30)
    
    # Create date index for predictions
    last_date = rolling_corr.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
    
    return {
        'predictions': predictions,
        'dates': future_dates,
        'current_correlation': rolling_corr.iloc[-1],
        'metrics': metrics,
        'confidence': confidence,
        'model_type': model_type
    }

if __name__ == "__main__":
    # Test the predictor
    print("Testing Correlation Predictor...")
    
    from rolling_correlation import load_asset_data
    
    # Load sample data
    spy = load_asset_data('../../data/raw/SPY_etf.csv')
    gold = load_asset_data('../../data/raw/GCF_commodity.csv')
    
    # Make prediction
    result = predict_correlation(spy, gold, window=90, days_ahead=30, model_type='linear')
    
    print(f"\nModel: {result['model_type']}")
    print(f"Current Correlation: {result['current_correlation']:.3f}")
    print(f"Predicted Correlation (30 days): {result['predictions'][-1]:.3f}")
    print(f"\nModel Performance:")
    print(f"  Train R²: {result['metrics']['train_r2']:.3f}")
    print(f"  Test R²: {result['metrics']['test_r2']:.3f}")
    print(f"\nPrediction Confidence:")
    print(f"  Mean Error: ±{result['confidence']['mean_error']:.3f}")