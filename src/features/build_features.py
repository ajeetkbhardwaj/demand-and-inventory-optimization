import os
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, dayofweek, dayofyear, when, lag, avg, stddev, last, lit
from pyspark.sql.window import Window
import duckdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_spark_session():
    """Initialize a local Spark session simulating Databricks."""
    return SparkSession.builder \
        .appName("FMCG_Feature_Engineering") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def build_features(spark, db_path: str, output_parquet: str):
    """
    Reads raw data from DuckDB (simulating Snowflake),
    uses PySpark to build scalable time-series features,
    and saves the engineered features to Parquet.
    """
    logging.info("Starting PySpark feature engineering...")
    
    # 1. Read from DuckDB using Arrow to PySpark
    con = duckdb.connect(db_path)
    
    sales_pdf = con.execute("SELECT * FROM raw_sales ORDER BY store_nbr, family, date").df()
    stores_pdf = con.execute("SELECT * FROM stores").df()
    oil_pdf = con.execute("SELECT * FROM oil").df()
    holidays_pdf = con.execute("SELECT * FROM holidays_events").df()
    transactions_pdf = con.execute("SELECT * FROM transactions").df()
    con.close()
    
    df = spark.createDataFrame(sales_pdf)
    stores_df = spark.createDataFrame(stores_pdf).withColumnRenamed("type", "store_type")
    oil_df = spark.createDataFrame(oil_pdf)
    holidays_df = spark.createDataFrame(holidays_pdf)
    transactions_df = spark.createDataFrame(transactions_pdf)
    
    # 1.5 Pre-process Oil data (Forward-fill on the small table to prevent global sorting on 3M rows)
    # Add a dummy partition key to satisfy Spark's optimizer and silence the WindowExec warning
    oil_df = oil_df.withColumn("_dummy", lit(1))
    window_oil = Window.partitionBy("_dummy").orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    oil_df = oil_df.withColumn("dcoilwtico", last("dcoilwtico", ignorenulls=True).over(window_oil)).drop("_dummy")
    
    # 2. Join Auxiliary Tables
    df = df.join(stores_df, on="store_nbr", how="left")
    df = df.join(oil_df, on="date", how="left").fillna({"dcoilwtico": 50.0})
    df = df.join(transactions_df, on=["date", "store_nbr"], how="left").fillna({"transactions": 0})
    
    national_holidays = holidays_df.filter(
        (col("locale") == "National") & (col("transferred") == False)
    ).select("date").distinct().withColumn("is_holiday", lit(1))
    
    df = df.join(national_holidays, on="date", how="left").fillna({"is_holiday": 0})
    
    # 3. Time-based features
    df = df.withColumn("month", month(col("date"))) \
           .withColumn("day_of_week", dayofweek(col("date"))) \
           .withColumn("day_of_year", dayofyear(col("date"))) \
           .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
           
    # 4. Define Window for Lag and Rolling features
    # Partition by store and product family, ordered by date
    windowSpec = Window.partitionBy("store_nbr", "family").orderBy("date")
    
    # 5. Lag Features (e.g., sales from 1 day ago, 7 days ago)
    df = df.withColumn("sales_lag_1", lag("sales", 1).over(windowSpec)) \
           .withColumn("sales_lag_7", lag("sales", 7).over(windowSpec)) \
           .withColumn("sales_lag_28", lag("sales", 28).over(windowSpec)) \
           .withColumn("transactions_lag_1", lag("transactions", 1).over(windowSpec)) \
           .withColumn("transactions_lag_7", lag("transactions", 7).over(windowSpec))
           
    # 6. Rolling Window Features (e.g., 7-day moving average and std dev)
    # Need a window spec for the past 7 days
    windowSpec_7 = windowSpec.rowsBetween(-7, -1)
    windowSpec_28 = windowSpec.rowsBetween(-28, -1)
    
    df = df.withColumn("rolling_mean_7", avg("sales").over(windowSpec_7)) \
           .withColumn("rolling_std_7", stddev("sales").over(windowSpec_7)) \
           .withColumn("rolling_mean_28", avg("sales").over(windowSpec_28))

    # Drop rows with nulls introduced by lags to keep dataset clean for ML
    df = df.dropna()

    logging.info(f"Writing engineered features to {output_parquet}...")
    # 6. Save engineered features back to storage (Parquet format simulates writing back to Snowflake/ADLS)
    df.write.mode("overwrite").parquet(output_parquet)
    
    logging.info("Feature engineering complete.")
    spark.stop()

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    db_path = os.path.join(project_dir, "data", "warehouse.duckdb")
    output_parquet = os.path.join(project_dir, "data", "processed", "features.parquet")
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    
    spark = get_spark_session()
    build_features(spark, db_path, output_parquet)
