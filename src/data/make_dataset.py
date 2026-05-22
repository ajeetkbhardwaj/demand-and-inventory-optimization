import os
import logging
import duckdb
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def store_in_duckdb(data_dir: str, db_path: str):
    """
    Reads the raw CSV files and stores them into DuckDB as Parquet-backed tables.
    This simulates a Snowflake ingestion process.
    """
    logging.info(f"Ingesting raw data into DuckDB at {db_path}...")
    train_csv = os.path.join(data_dir, 'train.csv')
    
    if not os.path.exists(train_csv):
        logging.error(f"File {train_csv} not found.")
        return
        
    con = duckdb.connect(db_path)
    
    # Create table from CSV
    logging.info("Ingesting main sales file train.csv into table raw_sales...")
    con.execute(f"CREATE TABLE IF NOT EXISTS raw_sales AS SELECT * FROM read_csv_auto('{train_csv}')")
    
    # Load auxiliary tables if they exist
    aux_files = {
        'stores': 'stores.csv',
        'oil': 'oil.csv',
        'holidays_events': 'holidays_events.csv',
        'transactions': 'transactions.csv'
    }
    
    for table_name, file_name in aux_files.items():
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            logging.info(f"Ingesting auxiliary file {file_name} into table {table_name}...")
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
            
    logging.info("All available data successfully ingested into DuckDB.")
    con.close()

if __name__ == "__main__":
    # Define paths
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = os.path.join(project_dir, "data", "raw")
    db_path = os.path.join(project_dir, "data", "warehouse.duckdb")
    
    os.makedirs(raw_data_dir, exist_ok=True)
    
    train_path = os.path.join(raw_data_dir, 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Kaggle data not found at {train_path}. Please ensure your Kaggle CSV files are placed in the data/raw directory.")
    else:
        logging.info("Found Kaggle data in data/raw. Proceeding to DuckDB ingestion.")
        
    # Ingest to simulated Data Warehouse (DuckDB)
    store_in_duckdb(raw_data_dir, db_path)
    logging.info("Data ingestion phase complete.")
