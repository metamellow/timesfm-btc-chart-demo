### Important Links

- **TimesFM Documentation**: https://github.com/google-research/timesfm  
- **yfinance Docs**: https://pypi.org/project/yfinance/  
- Google Blog: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/

---

# TimesFM Crypto Forecasting - VS Code Workflow

## Overview

This guide fetches Bitcoin prices (last 730 days) using `yfinance`, runs TimesFM v2.5 (PyTorch backend) for a 30-day forecast, detects trend switches, and generates a readable chart with red lines for switches, plus a text summary. Enhanced for better chart readability (wider layout, selective labels, inset zoom) and includes metrics to assess TimesFM’s contribution (statistics, baseline comparison). All installs are local.

## Prerequisites

- Windows 10/11 with Git Bash
- VS Code
- Python 3.12
- Git installed
- No GPU needed (CPU with PyTorch; GPU optional)

---

## PART 1: VS Code Setup

### Step 1: Verify Tools

```bash
python --version
git --version
```

### Step 2: Create Structure

```bash
mkdir timesfm-crypto-forecast
cd timesfm-crypto-forecast
mkdir scripts output
```

### Step 3: Create Environment

```bash
python -m venv venv
source venv/Scripts/activate
```

### Step 4: Clone TimesFM

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .[torch]
cd ..
```

### Step 5: Create Script

```bash
cat > scripts/forecast.py << EOF
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
EOF
```

### Step 6: Git Commit

```bash
git add .
git commit -m "Enhanced chart readability and TimesFM analysis"
git push
```

---

## PART 2: Terminal Setup

### Step 1: Install Dependencies

```bash
source venv/Scripts/activate
python -m pip install --upgrade pip
pip install yfinance pandas numpy matplotlib
echo "✅ All installations complete (local to venv)"
```

---

## PART 3: Terminal Run

### Step 1: Clean Up

```bash
rm -f output/btc_trend.png 2>/dev/null || true
rm -f run-log.md 2>/dev/null || true
echo "✅ Cleaned up old files"
```

### Step 2: Activate Environment

```bash
source venv/Scripts/activate
echo "✅ Virtual environment activated"
```

### Step 3: Run Forecast Script

```bash
echo "Running forecast..."
python scripts/forecast.py | tee run-output.txt
echo "✅ Forecast complete"
```

### Step 4: Create Run Log

```bash
cat > run-log.md << EOF
# Forecast Run Log - $(date)

## Summary
- Data Source: yfinance (BTC, 730 days)
- Forecast Horizon: 30 days
- Model: TimesFM v2.5 (PyTorch backend)
- Backend: CPU (GPU optional)

## Key Outputs
- Chart: output/btc_trend.png (historical + forecast + switches)
- Overall Trend: From run-output.txt
- Switches: Dates of trend flips

## Interpretation
- Green line: TimesFM prediction.
- Orange dashed: Naive (last price) baseline.
- Green shade: 80% confidence.
- Red lines: Trend switches.
- Top chart: Full view; bottom: 30-day zoom.

## Next Steps
1. View output/btc_trend.png
2. Check run-output.txt for stats
3. Adjust for other cryptos
EOF
echo "✅ Log saved to run-log.md"
```

---

## Troubleshooting Notes

1. **Chart Readability**: Wider figure (12x10), inset zoom, and selective switch labels reduce clutter. Adjust `figsize` or `switches[::2]` if needed.
2. **TimesFM Insight**: Stats (mean, variance, change) and naive baseline show model activity. High variance or significant deviation from naive suggests TimesFM is adding value.
3. **Warnings**: PyTorch TF32 and Hugging Face symlink warnings are safe (model loaded fine).
4. **GPU**: Test with `torch.cuda.is_available()` after installing CUDA PyTorch.
