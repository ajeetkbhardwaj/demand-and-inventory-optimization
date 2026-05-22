import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_lightgbm(parquet_path: str, model_dir: str):
    """
    Trains a LightGBM model on the engineered features.
    Simulates training a scalable ML model for multiple SKU/Stores.
    """
    logging.info(f"Loading features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Sort temporally for time-based split
    df = df.sort_values('date')
    
    # Convert identifiers to categorical so LightGBM learns store/SKU hierarchies natively
    categorical_cols = ['store_nbr', 'family', 'city', 'state', 'store_type', 'cluster', 'is_holiday']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Define features and target
    features = [
        'store_nbr', 'family', 'city', 'state', 'store_type', 'cluster',
        'onpromotion', 'month', 'day_of_week', 'day_of_year', 'is_weekend', 'is_holiday',
        'dcoilwtico',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_28',
        'transactions_lag_1', 'transactions_lag_7',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28'
    ]
    target = 'sales'
    
    # Time-based Train/Test Split (e.g., last 30 days for validation)
    split_date = df['date'].max() - pd.Timedelta(days=30)
    
    train = df[df['date'] <= split_date]
    valid = df[df['date'] > split_date]
    
    X_train, y_train = train[features], train[target]
    X_valid, y_valid = valid[features], valid[target]
    
    logging.info(f"Training LightGBM with {len(X_train)} rows and {len(X_valid)} validation rows.")
    
    # LightGBM setup
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # 1. Train Model 1: Mean Point Forecast (Tweedie Loss)
    # Tweedie is the SOTA objective for retail/FMCG because it natively handles 
    # both high-volume (Beverages) and low-volume/zero-inflated data (Automotive).
    params_point = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5, # 1.5 = Compound Poisson-Gamma (SOTA for extreme zero-inflation)
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127, # Increased so the global model can learn more complex per-SKU rules
        'min_data_in_leaf': 10, # Lowered from 20 to allow the AI to learn from rare, spiky events
        'max_bin': 511, # Increased from 255 to allow finer splits on numeric features (like rolling stats)
        'feature_fraction': 0.8,
        'seed': 42,
        'verbose': -1
    }
    
    logging.info("Training LightGBM Model 1 (Tweedie Mean Forecast)...")
    model_median = lgb.train(
        params_point,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 2. Train Model 2: 95th Quantile Forecast for AI-driven Safety Stock
    params_q95 = params_point.copy()
    params_q95['objective'] = 'quantile'
    params_q95['alpha'] = 0.95
    params_q95['metric'] = 'quantile'
    
    logging.info("Training LightGBM Model 2 (95th Quantile Forecast)...")
    model_q95 = lgb.train(
        params_q95,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Evaluation
    preds = model_median.predict(X_valid)
    preds = np.maximum(0, preds) # Ensure no negative predictions
    
    mae = mean_absolute_error(y_valid, preds)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    r2 = r2_score(y_valid, preds)
    
    # Mathematically robust SMAPE to handle Zero-Sales days
    denominator = (np.abs(y_valid) + np.abs(preds)) / 2.0
    smape = np.mean(np.where(denominator == 0, 0.0, np.abs(y_valid - preds) / denominator)) * 100
    
    logging.info(f"Validation MAE: {mae:.2f}")
    logging.info(f"Validation RMSE: {rmse:.2f}")
    logging.info(f"Validation R² (Accuracy): {r2:.3f}")
    logging.info(f"Validation (S)MAPE: {smape:.2f}%")
    
    # Save Model
    model_path = os.path.join(model_dir, 'lgb_model.pkl')
    joblib.dump(model_median, model_path)  # Main model used by the UI Explainability
    logging.info(f"Model saved to {model_path}")
    
    model_q95_path = os.path.join(model_dir, 'lgb_model_q95.pkl')
    joblib.dump(model_q95, model_q95_path)
    logging.info(f"Quantile Model saved to {model_q95_path}")
    
    # In a real environment, you might also train statsmodels SARIMA here per SKU
    # But for millions of rows, LightGBM on lags is vastly more efficient globally.
    
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    parquet_path = os.path.join(project_dir, "data", "processed", "features.parquet")
    model_dir = os.path.join(project_dir, "src", "models")
    
    if not os.path.exists(parquet_path):
        logging.error("Features Parquet not found. Please run build_features.py first.")
    else:
        train_lightgbm(parquet_path, model_dir)
