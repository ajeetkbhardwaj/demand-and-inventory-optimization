# End-to-End Retail Demand Planning & Inventory Optimization Pipeline

An applied mathematical and engineering-focused demand forecasting pipeline built for supply chain optimization. Designed to simulate enterprise tools (Databricks, Snowflake) using scalable open-source equivalents (Local PySpark, DuckDB) and modern AI techniques (LightGBM, NVIDIA NIM API).

This project bridges the gap between advanced Machine Learning and Supply Chain Operations by translating daily demand predictions directly into **actionable inventory decisions**—specifically, dynamically optimized Safety Stock targets based on target Service Levels.

## Architecture & Tech Stack

* **Data Warehouse Simulation**: `DuckDB` acting as a local, fast analytical store (simulating Snowflake/BigQuery).
* **Scalable Processing**: `PySpark` for scalable time-series transformations and rolling window features.
* **Predictive Modeling**: `LightGBM` Ensemble (Median Point Forecast + 95th Quantile Risk Forecast) for non-linear lag/promo modeling and dynamic variance.
* **Inventory Math**: Probabilistic forecasting and Safety Stock optimization using SOTA Stochastic Lead Time equations (`scipy.stats`).
* **UI / Dashboard**: High-performance `Streamlit` application for Planner Business Intelligence.
* **Generative AI Co-Pilot**: NVIDIA NIM API (`gemma-3` multimodal LLM) for automated data storytelling and visual chart analysis.
* **CI/CD & Deployment**: Dockerized container deployment to Hugging Face Spaces via GitHub Actions.

## Project Structure

```text
├── README.md
├── requirements.txt
├── app
│   └── main.py                     # Streamlit planner dashboard
├── data
│   ├── raw                         # Raw Kaggle data files
│   └── processed                   # DuckDB warehouse and Parquet files
├── notebooks
│   └── 01_mathematical_foundations.ipynb  # Math whitepaper
└── src
    ├── data
    │   └── make_dataset.py         # Data ingestion into DuckDB
    ├── features
    │   └── build_features.py       # PySpark time-series engineering
    └── models
        ├── train_model.py          # LightGBM training
        └── predict_and_optimize.py # Safety Stock & FVA calculations
├── .github/workflows/
│   └── deploy.yml                  # CI/CD pipeline to Hugging Face
├── Dockerfile                      # Container configuration
└── .dockerignore                   # Ignores large raw files for deployment
```

## Setup Instructions

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install -U requests kaleido  # Required for NVIDIA LLM calls and Plotly image exports
   ```
2. **Run the Pipeline end-to-end:**

   ```bash
   python src/data/make_dataset.py
   python src/features/build_features.py
   python src/models/train_model.py
   python src/models/predict_and_optimize.py
   ```
3. **Launch the Dashboard:**

   ```bash
   streamlit run app/main.py
   ```

   *(Note: To use the AI Co-Pilot data storytelling feature, you will need to enter your NVIDIA API key in the sidebar of the dashboard).*
4. Hugging Face Deployment (Docker) : This project is equipped with a Dockerfile and GitHub Actions workflow for automatic deployment to Hugging Face Spaces. 1. Create a Blank Docker Space on Hugging Face. 2. Set your Hugging Face Token as a repository secret (`HF_TOKEN`) in GitHub.  3. Push to `main`. The CI/CD pipeline will automatically build and deploy the planner dashboard to the web.

## Engineering Highlights & Mathematical Core

This pipeline overcomes the most difficult challenges in enterprise demand forecasting by graduating from simple point-forecasting to a **Multi-Model Probabilistic Forecaster**:

### 1. Handling Intermittent Demand (Tweedie Regression)
Training a single global model across thousands of SKUs traditionally fails because slow-moving items (high zero-inflation) are treated as noise by standard loss functions (MAE/MSE). We utilize a **Tweedie Loss Objective** (Compound Poisson-Gamma distribution) in LightGBM. This SOTA approach (popularized by the Kaggle M5 competition) allows a single model to flawlessly scale across both high-volume continuous demand (e.g., Beverages) and erratic, intermittent spikes (e.g., Automotive).

### 2. Dynamic AI Variance for Safety Stock
Standard ERPs use static historical standard deviations to calculate safety stock, treating the uncertainty of a standard Tuesday exactly the same as Black Friday. Our dual-model AI ensemble predicts both the *Expected Demand* and the *95th Quantile Risk*. 

We dynamically extract the daily standard deviation ($\sigma_D$) strictly from the spread between the median and quantile predictions ($\sigma_D \approx \frac{Q95 - Median}{1.645}$). This forces Safety Stock to automatically inflate prior to high-risk events and deflate during stable periods to free up working capital.

### 3. Stochastic Lead Time Inventory Math
We combine the dynamic AI variance with stochastic lead times to formulate optimized safety stock ($SS$) and Order-Up-To levels for any chosen Service Level ($Z$):

$$
SS = Z \times \sqrt{(L \times \sigma_D^2) + (\mu_D^2 \times \sigma_L^2)}
$$

### 4. Forecast Value Add (FVA) & AI Explainability
To solve the "Black Box" problem and build planner trust:
* **FVA Tracking:** The pipeline mathematically proves its worth by benchmarking ML RMSE and MAE against naive baseline metrics (e.g., "yesterday's sales").
* **Multimodal LLM Co-Pilot:** We integrate the **NVIDIA NIM API (Gemma-3)** to act as an automated Data Scientist. The LLM analyzes the dashboard's forecast metrics, feature importance (SHAP/Gain), and visual charts to generate plain-English executive summaries and actionable inventory mitigation strategies.
