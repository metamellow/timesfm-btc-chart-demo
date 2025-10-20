import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import timesfm
import sys
import torch
from datetime import datetime

# Step 1: Fetch Data
def fetch_data():
    try:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
        data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError("No data fetched")
        df = pd.DataFrame({
            'unique_id': 'BTC',
            'ds': data.index,
            'y': data['Close'].values.flatten()
        })
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna().set_index('ds').resample('D').ffill().reset_index()
        print(f"Data fetched from yfinance (730 days, {len(df)} points) successfully.")
        return df
    except Exception as e:
        print(f"yfinance failed: {e}")
        sys.exit(1)

df = fetch_data()
print(df.tail())

# Step 2: Initialize TimesFM
torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# Step 3: Run Forecast
inputs = [df['y'].values[-512:].astype(np.float32)]
horizon = 30
point_forecast, quantile_forecast = model.forecast(inputs=inputs, horizon=horizon)

# Create forecast DataFrame
forecast_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
forecast_df = pd.DataFrame({
    'ds': forecast_dates,
    'timesfm': point_forecast[0],
    'timesfm-q-0.1': quantile_forecast[0, :, 0],
    'timesfm-q-0.9': quantile_forecast[0, :, -1],
})
print(forecast_df.head())

# Step 4: Baseline Comparison (Naive Forecast: Last Price)
naive_forecast = np.full(horizon, df['y'].iloc[-1])
naive_df = pd.DataFrame({'ds': forecast_dates, 'naive': naive_forecast})

# Step 5: Detect Trends and Switches
current_price = df['y'].iloc[-1]
forecast_prices = forecast_df['timesfm'].values
overall_trend = 'Bullish' if forecast_prices[-1] > current_price else 'Bearish'

differences = forecast_prices[1:] - forecast_prices[:-1]
daily_trends = ['Bullish' if diff > 0 else 'Bearish' for diff in differences]
switches = [(forecast_df['ds'].iloc[i+1], daily_trends[i]) for i in range(1, len(daily_trends)) if daily_trends[i] != daily_trends[i-1]]

# Print TimesFM Contribution
print(f"\nTimesFM Forecast Stats:")
print(f"Mean Forecast: {forecast_prices.mean():.2f}")
print(f"Variance: {np.var(forecast_prices):.2f}")
print(f"Change from Current: {forecast_prices[-1] - current_price:.2f}")
print(f"Naive Forecast (Last Price): {naive_forecast[0]:.2f}")

# Step 6: Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(df['ds'], df['y'], label="Historical BTC Price")
ax1.plot(forecast_df['ds'], forecast_df['timesfm'], label="TimesFM Forecast", color="green")
ax1.plot(naive_df['ds'], naive_df['naive'], label="Naive Forecast (Last Price)", color="orange", linestyle="--")
ax1.fill_between(forecast_df['ds'], forecast_df['timesfm-q-0.1'], forecast_df['timesfm-q-0.9'], color="green", alpha=0.3, label="80% Prediction Interval")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax1.set_title(f"BTC Price Forecast with Trend Switches (06:34 PM AWST, Oct 17, 2025)")
ax1.legend()

# Plot switches on main chart (every other for clarity)
for i, (date, trend) in enumerate(switches[::2]):  # Every other switch
    ax1.axvline(x=date, color='red', linestyle='--')
    ax1.text(date, forecast_prices.mean(), f"Switch to {trend}", rotation=90, va='center')

# Inset Zoom on Forecast Period
ax2.plot(forecast_df['ds'], forecast_df['timesfm'], label="TimesFM Forecast", color="green")
ax2.plot(naive_df['ds'], naive_df['naive'], label="Naive Forecast", color="orange", linestyle="--")
ax2.fill_between(forecast_df['ds'], forecast_df['timesfm-q-0.1'], forecast_df['timesfm-q-0.9'], color="green", alpha=0.3)
for date, trend in switches:
    ax2.axvline(x=date, color='red', linestyle='--', alpha=0.5)
ax2.set_title("Zoom: 30-Day Forecast with Switches")
ax2.legend()
ax2.set_xlim(forecast_df['ds'].iloc[0], forecast_df['ds'].iloc[-1])

plt.tight_layout()
plt.savefig('output/btc_trend.png')
plt.show()
print("Chart saved to output/btc_trend.png")
