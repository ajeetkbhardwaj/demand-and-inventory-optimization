import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from pathlib import Path
import os
import joblib
import requests
import json
import base64

st.set_page_config(page_title="Demand Planning Workspace", layout="wide", page_icon="📦")

# Add some custom CSS to make it look premium
st.markdown("""
<style>
    .reportview-container {
        background: #fafafa;
    }
    h1, h2, h3 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

project_dir = Path(__file__).resolve().parents[1]

@st.cache_data
def load_data():
    data_path = os.path.join(project_dir, "data", "processed", "analytical_results.parquet")
    
    if not os.path.exists(data_path):
        return pd.DataFrame() # Return empty if not run yet
        
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

st.title("Demand Planning & Inventory Optimization")
st.markdown("Interactive tool for supply chain planners to analyze forecasts, biases, and optimize safety stock.")

df = load_data()

if df.empty:
    st.warning("Analytical results not found. Please run the pipeline (make_dataset.py -> build_features.py -> train_model.py -> predict_and_optimize.py).")
else:
    # Sidebar Filters
    st.sidebar.header("Filters")
    store_list = sorted(df['store_nbr'].unique())
    family_list = sorted(df['family'].unique())
    
    selected_store = st.sidebar.selectbox("Select Store", store_list)
    selected_family = st.sidebar.selectbox("Select Product Family (SKU)", family_list)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Co-Pilot LLM")
    #nv_api_key = st.sidebar.text_input("NVIDIA API Key", type="password", value=os.environ.get("NVIDIA_API_KEY", ""), help="Required to generate live LLM summaries. Securely loaded from environment if available.")
    
    env_api_key = os.environ.get("NVIDIA_API_KEY", "")
    if env_api_key:
        nv_api_key = env_api_key
        st.sidebar.success("🔑 NVIDIA API Key securely loaded from environment.")
    else:
        nv_api_key = st.sidebar.text_input("NVIDIA API Key", type="password", help="Required to generate live LLM summaries.")
    
    # Filter Data
    filtered_df = df[(df['store_nbr'] == selected_store) & (df['family'] == selected_family)].copy()
    filtered_df = filtered_df.sort_values('date')
    
    # KPIs
    st.markdown("### Model Performance KPIs")
    
    mae = filtered_df['error'].abs().mean()
    bias = filtered_df['error'].mean()
    avg_sales = filtered_df['sales'].mean()
    rmse = np.sqrt((filtered_df['error'] ** 2).mean())
    
    ss_res = (filtered_df['error'] ** 2).sum()
    ss_tot = ((filtered_df['sales'] - avg_sales) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Safe division to prevent zero-sales errors
    bias_pct = bias / avg_sales if avg_sales != 0 and not pd.isna(avg_sales) else 0.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("Model Accuracy (R²)", f"{r2:.2f}")
    with col4:
        st.metric("Forecast Bias (Mean Error)", f"{bias:.2f}", delta=f"{bias_pct:.1%}", delta_color="inverse")
    with col5:
        fva_naive_mae = (filtered_df['sales'] - filtered_df['sales'].shift(1)).abs().mean()
        fva = (fva_naive_mae - mae) / fva_naive_mae if fva_naive_mae != 0 and not pd.isna(fva_naive_mae) else 0.0
        st.metric("Forecast Value Add (FVA)", f"{fva:.1%}")

    # Forecast vs Actual Chart
    st.markdown("### Forecast vs Actuals")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['sales'], mode='lines', name='Actual Sales', line=dict(color='#3B82F6', width=2)))
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['forecast_sales'], mode='lines', name='Forecast', line=dict(color='#F59E0B', width=2, dash='dash')))
    
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    st.plotly_chart(fig, width='stretch')

    # Model Generalization & Error Distribution
    st.markdown("---")
    st.markdown("### Model Generalization & Error Distribution")
    st.markdown("A well-generalized model will have normally distributed errors centered around zero. Skewed errors indicate bias, while a tight Actual vs Predicted spread shows strong accuracy across demand scales.")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        fig_hist = px.histogram(filtered_df, x='error', nbins=50, title="Residuals (Error) Distribution", opacity=0.75, color_discrete_sequence=['#8B5CF6'])
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
        fig_hist.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, width="stretch")
        
    with dist_col2:
        fig_scatter = px.scatter(filtered_df, x='sales', y='forecast_sales', opacity=0.6, title="Actual vs Predicted Sales", color_discrete_sequence=['#10B981'])
        max_val = max(filtered_df['sales'].max(), filtered_df['forecast_sales'].max())
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
        fig_scatter.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, width="stretch")

    # Inventory Optimization Simulator
    st.markdown("---")
    st.markdown("### Inventory Optimization Simulator")
    st.markdown("Adjust the desired service level to see the impact on Safety Stock.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        target_service_level = st.slider("Target Cycle Service Level (%)", min_value=80, max_value=99, value=95, step=1)
        lead_time = st.number_input("Lead Time (Days)", min_value=1, max_value=30, value=7)
        lead_time_sd = st.number_input("Lead Time Std Dev (Days)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
        
        # Calculate new safety stock based on math
        sigma_1 = filtered_df['error'].std()
        mu_d = filtered_df['forecast_sales'].mean()
        
        sigma_L = np.sqrt((lead_time * (sigma_1 ** 2)) + ((mu_d ** 2) * (lead_time_sd ** 2)))
        z_score = norm.ppf(target_service_level / 100.0)
        simulated_safety_stock = z_score * sigma_L
        
        st.info(f"**Math Breakdown:**\n\n$\\sigma_D$ (Daily Error Std Dev): {sigma_1:.2f}\n\n$\\mu_D$ (Avg Daily Demand): {mu_d:.2f}\n\n$Z$-score for {target_service_level}%: {z_score:.2f}\n\n$\\sigma_L$ (Stochastic Lead Time Volatility): {sigma_L:.2f}\n\n**Calculated SS = {simulated_safety_stock:.1f} units**")

    with sim_col2:
        # Plot tradeoff curve
        service_levels = np.linspace(0.80, 0.999, 100)
        z_scores = norm.ppf(service_levels)
        ss_curve = z_scores * sigma_L
        
        tradeoff_df = pd.DataFrame({'Service Level': service_levels * 100, 'Safety Stock Required': ss_curve})
        
        fig2 = px.line(tradeoff_df, x='Service Level', y='Safety Stock Required', title="Safety Stock vs. Service Level Tradeoff")
        
        # Add point for current selection
        fig2.add_trace(go.Scatter(x=[target_service_level], y=[simulated_safety_stock], mode='markers', marker=dict(color='red', size=10), name='Selected Policy'))
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, width='stretch')
        
    # Model Explainability
    st.markdown("---")
    st.markdown("### Model Explainability (Feature Importance)")
    st.markdown("What external and internal factors is the AI using to drive these forecasts?")
    
    model_path = os.path.join(project_dir, "src", "models", "lgb_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Get feature importance based on Information Gain
        importance = pd.DataFrame({
            'Feature': model.feature_name(),
            'Importance': model.feature_importance(importance_type='gain')
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig3 = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Top 10 AI Demand Drivers (Information Gain)")
        fig3.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, width="stretch")
        
        # Automated SKU-Level Business Report (Data Storytelling)
        top_3_features = ", ".join([f"**{f}**" for f in importance.tail(3)['Feature'].tolist()[::-1]])
        bias_text = "over-forecast" if bias < 0 else "under-forecast" if bias > 0 else "accurately predict"
        
        st.markdown("---")
        col_rep_title, col_rep_btn = st.columns([3, 1])
        with col_rep_title:
            st.markdown(f"### 📝 AI Co-Pilot Business Report: {selected_family} at Store {selected_store}")
        with col_rep_btn:
            generate_llm = st.button("✨ Generate with NVIDIA LLM", use_container_width=True)

        if generate_llm:
            if not nv_api_key:
                st.error("🔑 Please enter your NVIDIA API Key in the sidebar first!")
            else:
                report_placeholder = st.empty()
                
                # 1. Build the LLM prompt dynamically
                prompt = f"""You are an elite Supply Chain AI Data Scientist presenting to C-Suite executives. Analyze the metrics below and provide deep, actionable business insights rather than just repeating the numbers.

Context Data:
- Store: {selected_store}
- Product Family: {selected_family}
- Average Daily Sales: {avg_sales:.1f} units
- Mean Absolute Error (MAE): {mae:.2f} units
- Root Mean Squared Error (RMSE): {rmse:.2f} units
- Model Accuracy (R²): {r2:.2f}
- Forecast Bias: {bias_pct:.1%} (tending to {bias_text})
- Forecast Value Add (FVA): {fva:.1%}
- Recommended Safety Stock (at {target_service_level}% Service Level): {simulated_safety_stock:.1f} units
- Top Demand Drivers identified by AI: {top_3_features}

Instructions:
Write a highly professional, data-driven 3-section executive report. You MUST structure your response exactly like this:
### 1. Executive Summary\nWrite a sharp, strategic paragraph summarizing the demand predictability and forecasting performance. If an image is provided, analyze the visual seasonality or volatility in the forecast vs. actuals chart.\n\n### 2. Inventory & Risk Mitigation\nProvide 3 bullet points. Explain the financial and operational impact of achieving a {fva:.1%} FVA. Detail how holding {simulated_safety_stock:.1f} units of Safety Stock mitigates stockout risk at a {target_service_level}% service level.\n\n### 3. Drivers of Demand\nBriefly explain the underlying physics of this SKU's demand. Why do the top drivers ({top_3_features}) matter for this specific product family? Provide a hypothetical business action the planner could take based on these drivers."""
                
                # 1.5 Convert Plotly chart to Base64 Image for the Multimodal LLM
                try:
                    # Exporting Plotly figures requires the 'kaleido' package
                    fig_bytes = fig.to_image(format="png", width=800, height=400)
                    fig_b64 = base64.b64encode(fig_bytes).decode('utf-8')
                    
                    message_content = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fig_b64}"}}
                    ]
                except ValueError:
                    # Graceful fallback to text-only if kaleido isn't installed
                    message_content = [{"type": "text", "text": prompt}]
                    st.warning("💡 Tip: To pass charts to the AI, run `pip install -U kaleido` in your terminal.")
                except Exception:
                    message_content = [{"type": "text", "text": prompt}]
                
                # 2. Setup the NVIDIA API call
                invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {nv_api_key}",
                    "Accept": "text/event-stream"
                }
                payload = {
                    "model": "google/gemma-3n-e4b-it",
                    "messages": [{"role": "user", "content": message_content}],
                    "max_tokens": 800,
                    "temperature": 0.35,
                    "top_p": 0.70,
                    "stream": True
                }
                
                # 3. Stream the output natively to the UI
                try:
                    response = requests.post(invoke_url, headers=headers, json=payload, stream=True)
                    response.raise_for_status()
                    
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            line_decoded = line.decode("utf-8")
                            if line_decoded.startswith("data: "):
                                data_str = line_decoded.replace("data: ", "")
                                if data_str.strip() != "[DONE]":
                                    try:
                                        data_json = json.loads(data_str)
                                        content = data_json["choices"][0]["delta"].get("content", "")
                                        full_response += content
                                        report_placeholder.info(full_response + "▌")
                                    except json.JSONDecodeError:
                                        pass
                    report_placeholder.info(full_response)
                except Exception as e:
                    st.error(f"API Error: {e}")
        
