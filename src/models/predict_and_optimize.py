import os
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_forecasts_and_inventory(parquet_path: str, model_path: str, output_path: str):
    """
    Applies the trained LightGBM model to generate point forecasts.
    Then, applies mathematical inventory optimization:
    Safety Stock = Z_alpha * sigma_L
    """
    logging.info("Loading ensemble models and validation data...")
    model_median = joblib.load(model_path)
    model_q95 = joblib.load(model_path.replace('.pkl', '_q95.pkl'))
    df = pd.read_parquet(parquet_path)
    
    # Ensure categories match the trained model
    categorical_cols = ['store_nbr', 'family', 'city', 'state', 'store_type', 'cluster', 'is_holiday']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    features = [
        'store_nbr', 'family', 'city', 'state', 'store_type', 'cluster',
        'onpromotion', 'month', 'day_of_week', 'day_of_year', 'is_weekend', 'is_holiday',
        'dcoilwtico',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_28',
        'transactions_lag_1', 'transactions_lag_7',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28'
    ]
    
    # Predict on the entire dataset (or just the latest partition for future)
    # For this exercise, we predict on historical to compute residuals
    df['forecast_sales'] = model_median.predict(df[features])
    df['forecast_sales'] = np.maximum(0, df['forecast_sales']) # No negative sales
    df['forecast_q95'] = np.maximum(0, model_q95.predict(df[features]))
    
    # 1. Compute Residuals & Variance
    df['error'] = df['sales'] - df['forecast_sales']
    
    # Calculate Forecast Value Add (FVA) against naive baseline (lag_1)
    df['naive_error'] = df['sales'] - df['sales_lag_1']
    mae_model = df['error'].abs().mean()
    rmse_model = np.sqrt((df['error'] ** 2).mean())
    mae_naive = df['naive_error'].abs().mean()
    rmse_naive = np.sqrt((df['naive_error'] ** 2).mean())
    
    fva = (mae_naive - mae_model) / mae_naive if mae_naive > 0 else 0.0
    fva_rmse = (rmse_naive - rmse_model) / rmse_naive if rmse_naive > 0 else 0.0
    logging.info(f"Forecast Value Add (FVA) against naive lag_1: {fva:.1%}")
    logging.info(f"Global Model RMSE: {rmse_model:.2f} (Naive: {rmse_naive:.2f}, FVA RMSE: {fva_rmse:.1%})")
    
    # 2. Inventory Math (Safety Stock)
    logging.info("Calculating Safety Stock via Service Level math...")
    # Assume Lead Time L = 7 days for the supply chain
    lead_time_days = 7
    lead_time_sd = 2  # Standard deviation of lead time in days (Stochastic fulfillment)
    
    # DYNAMIC AI VARIANCE: Use the spread between the 95th quantile and the median
    # instead of a static historical std dev. This means promotions/holidays automatically 
    # trigger higher Safety Stock buffers!
    # Z-score for 95% is ~1.645, so sigma_1 approx (q95 - median) / 1.645
    df['dynamic_daily_buffer'] = df['forecast_q95'] - df['forecast_sales']
    df['sigma_1'] = df['dynamic_daily_buffer'] / 1.645
    df['sigma_1'] = df['sigma_1'].fillna(0) # Safety check
    
    # Calculate average daily demand (mu_D) per SKU/Store
    demand_stats = df.groupby(['store_nbr', 'family'], observed=True)['forecast_sales'].mean().reset_index()
    demand_stats.rename(columns={'forecast_sales': 'mu_d'}, inplace=True)
    
    # Join variance back to predictions
    df = df.merge(demand_stats, on=['store_nbr', 'family'], how='left')
    
    # SOTA Math: Stochastic Lead Time Safety Stock Calculation
    df['sigma_L'] = np.sqrt((lead_time_days * (df['sigma_1'] ** 2)) + ((df['mu_d'] ** 2) * (lead_time_sd ** 2)))
    
    # Safety Stock for 95% Cycle Service Level (Z_0.95 approx 1.645)
    z_95 = norm.ppf(0.95)
    df['safety_stock_95'] = z_95 * df['sigma_L']
    
    # Order Up To Level (S) = Forecasted Lead Time Demand + Safety Stock
    # For simplicity, Lead Time Demand = forecast_sales * lead_time_days (assuming flat demand over L)
    df['order_up_to_level'] = (df['forecast_sales'] * lead_time_days) + df['safety_stock_95']
    
    logging.info(f"Saving final analytical dataset to {output_path}...")
    # Save the output for the Streamlit dashboard
    df[['date', 'store_nbr', 'family', 'sales', 'forecast_sales', 'error', 'safety_stock_95', 'order_up_to_level']].to_parquet(output_path)
    logging.info("Optimization complete.")

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    parquet_path = os.path.join(project_dir, "data", "processed", "features.parquet")
    model_path = os.path.join(project_dir, "src", "models", "lgb_model.pkl")
    output_path = os.path.join(project_dir, "data", "processed", "analytical_results.parquet")
    
    generate_forecasts_and_inventory(parquet_path, model_path, output_path)
