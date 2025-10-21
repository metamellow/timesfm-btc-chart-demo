import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
import sys
import torch
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Step 1: Fetch Data
def fetch_data():
    try:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
        data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError("No data fetched")
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date').resample('D').ffill().reset_index()
        print(f"Data fetched from yfinance (730 days, {len(data)} points) successfully.")
        return data
    except Exception as e:
        print(f"yfinance failed: {e}")
        sys.exit(1)

data = fetch_data()
print(data.tail())

# Preserve Date as a dedicated series before any column manipulation
if 'Date' in data.columns:
    date_series = pd.to_datetime(data['Date'])
else:
    date_series = pd.to_datetime(data.index.to_series().reset_index(drop=True))

# Step 2: Initialize TimesFM
torch.set_float32_matmul_precision("high")
model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(ForecastConfig(max_context=512, max_horizon=90))

# Step 3: Run Forecast
# Normalize potential MultiIndex columns from yfinance to single-level columns
def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        level1_values = df.columns.get_level_values(1)
        unique_level1 = list(dict.fromkeys(level1_values))  # preserve order
        if len(set(level1_values)) == 1:
            df = df.xs(unique_level1[0], axis=1, level=1, drop_level=True)
        else:
            target = 'BTC-USD' if 'BTC-USD' in set(level1_values) else unique_level1[0]
            df = df.xs(target, axis=1, level=1, drop_level=True)
    return df

data = normalize_yfinance_columns(data)

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')

close_series = to_numeric_series(data['Close'])
open_series = to_numeric_series(data['Open'])
high_series = to_numeric_series(data['High'])
low_series = to_numeric_series(data['Low'])
volume_series = to_numeric_series(data['Volume'])

inputs = [close_series.values[-512:].astype(np.float32)]  # 1D array
horizon = 90
print(f"Input shape: {inputs[0].shape}")

try:
    point_forecast, quantile_forecast = model.forecast(inputs=inputs, horizon=horizon)
    print(f"Point forecast shape: {point_forecast.shape}")
    print(f"Quantile forecast shape: {quantile_forecast.shape}")
    # Support both API shapes:
    # point_forecast: (B, C, H) or (B, H)
    if point_forecast.ndim == 3:
        forecast_values = point_forecast[0, -1, :]
    elif point_forecast.ndim == 2:
        forecast_values = point_forecast[0, :]
    else:
        raise ValueError(f"Unexpected point_forecast ndim: {point_forecast.ndim}")

    # quantile_forecast: (B, C, H, Q) or (B, H, Q)
    if quantile_forecast.ndim == 4:
        quantile_low = quantile_forecast[0, -1, :, 0]
        quantile_high = quantile_forecast[0, -1, :, -1]
    elif quantile_forecast.ndim == 3:
        quantile_low = quantile_forecast[0, :, 0]
        quantile_high = quantile_forecast[0, :, -1]
    else:
        raise ValueError(f"Unexpected quantile_forecast ndim: {quantile_forecast.ndim}")
except Exception as e:
    print(f"Forecast failed: {e}")
    sys.exit(1)

# Create forecast DataFrame
last_date = date_series.iloc[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Close': forecast_values,
    'q-0.1': quantile_low,
    'q-0.9': quantile_high,
})

# Step 4: Baseline (Linear Regression on last 90 days)
recent_close = close_series.iloc[-90:]
X = np.arange(len(recent_close)).reshape(-1, 1)
y = recent_close.values
lr = LinearRegression()
lr.fit(X, y)
baseline_X = np.arange(len(recent_close), len(recent_close) + horizon).reshape(-1, 1)
baseline_forecast = lr.predict(baseline_X).ravel()
baseline_df = pd.DataFrame({'Date': forecast_dates, 'Baseline': baseline_forecast})

# Step 5: Detect Trends and Switches
# Historical trends
historical_diff = close_series.diff().dropna()
historical_trends = ['Bullish' if d > 0 else 'Bearish' for d in historical_diff]
historical_switches = []
for i in range(1, len(historical_trends)):
    if historical_trends[i] != historical_trends[i-1]:
        historical_switches.append((date_series.iloc[i], historical_trends[i]))

# Forecast trends
forecast_diff = forecast_df['Close'].diff().dropna()
forecast_trends = ['Bullish' if d > 0 else 'Bearish' for d in forecast_diff]
forecast_switches = []
for i in range(1, len(forecast_trends)):
    if forecast_trends[i] != forecast_trends[i-1]:
        forecast_switches.append((forecast_df['Date'].iloc[i], forecast_trends[i]))

current_price = float(close_series.iloc[-1])
forecast_prices = forecast_df['Close'].values
overall_trend = 'Bullish' if forecast_prices[-1] > current_price else 'Bearish'

# Print Stats
print("\nTimesFM Forecast Stats:")
print(f"Mean: {forecast_prices.mean():.2f}")
print(f"Variance: {np.var(forecast_prices):.2f}")
print(f"Change from Current: {forecast_prices[-1] - current_price:.2f}")
print(f"Baseline Mean: {baseline_forecast.mean():.2f}")
print(f"Overall Trend (30 days): {overall_trend}")
print("Historical Trend Switches (Date, Trend):")
for date, trend in historical_switches[-5:]:  # Last 5 switches
    print(f"{date.date()}: {trend}")
print("Forecast Trend Switches (Date, Trend):")
for date, trend in forecast_switches:
    print(f"{date.date()}: {trend}")

# Step 6: Add Indicators
data['MA50'] = close_series.rolling(window=50).mean()
data['MA200'] = close_series.rolling(window=200).mean()

# Step 7: Visualize (clean, modern style)
fig, ax1 = plt.subplots(figsize=(16, 8), sharex=True)

# Price line (cleaner than dense candlesticks)
ax1.plot(date_series, close_series, label='Close', color='#1f77b4', linewidth=1.5)

# MAs
ax1.plot(date_series, data['MA50'], label='50-day MA', color='orange', linewidth=1.5)
ax1.plot(date_series, data['MA200'], label='200-day MA', color='purple', linewidth=1.5)

# Forecast: median line + channel, with background shade for forecast window
forecast_start = forecast_df['Date'].iloc[0]
forecast_end = forecast_df['Date'].iloc[-1]
ax1.axvspan(forecast_start, forecast_end, color='#2ca02c', alpha=0.06, linewidth=0)
ax1.plot(forecast_df['Date'], forecast_df['Close'], label='Forecast median', color='#2ca02c', linewidth=2.5)
ax1.scatter(forecast_df['Date'].iloc[::7], forecast_df['Close'].iloc[::7], color='#2ca02c', s=16, zorder=3)
ax1.fill_between(forecast_df['Date'], forecast_df['q-0.1'], forecast_df['q-0.9'], color="#2ca02c", alpha=0.25, label="Forecast channel (10–90%)")
ax1.axvline(forecast_start, color='#2ca02c', linestyle='--', alpha=0.6)
ax1.text(forecast_start, forecast_prices.mean(), ' Forecast start', color='#2ca02c', va='bottom', ha='left')

# Volume (lighter styling)
ax2 = ax1.twinx()
ax2.bar(date_series, volume_series, color='gray', alpha=0.15, label='Volume')
ax2.set_ylabel('Volume')
ax2.legend(loc='upper left')

""" Removed mini-chart inset to avoid covering recent data """

ax1.set_title("BTC Forecast (clean view)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True)
# Historical price channel (rolling 60d 10–90%)
rolling_window = 60
hist_low = close_series.rolling(rolling_window, min_periods=rolling_window).quantile(0.10)
hist_high = close_series.rolling(rolling_window, min_periods=rolling_window).quantile(0.90)
ax1.fill_between(date_series, hist_low, hist_high, color="#888888", alpha=0.12, label="Historical channel (10–90%)")

# Limit visible x-range to last 180 days and include forecast horizon end
history_days = 180
history_start = pd.to_datetime(last_date) - pd.Timedelta(days=history_days)
x_start = max(history_start, date_series.iloc[0])
x_end = forecast_df['Date'].iloc[-1]
ax1.set_xlim(x_start, x_end)

# Tidy y-limits to visible data range for readability
visible_mask_hist = (date_series >= x_start)
visible_mask_fore = (forecast_df['Date'] >= x_start)
visible_values = np.concatenate([
    close_series[visible_mask_hist].values,
    data['MA50'][visible_mask_hist].dropna().values,
    data['MA200'][visible_mask_hist].dropna().values,
    forecast_df['q-0.1'][visible_mask_fore].values,
    forecast_df['q-0.9'][visible_mask_fore].values,
])
if visible_values.size:
    ymin = np.nanmin(visible_values)
    ymax = np.nanmax(visible_values)
    span = ymax - ymin
    ax1.set_ylim(ymin - 0.05 * span, ymax + 0.05 * span)

# Annotate expected change at horizon
final_price = float(forecast_df['Close'].iloc[-1])
change_pct = (final_price - current_price) / current_price * 100.0
ax1.text(forecast_end, final_price, f"  {change_pct:+.2f}% in {horizon}d", color='#2ca02c', va='center', ha='left')
plt.tight_layout()
plt.savefig('output/btc_trend.png')
plt.show()
print("Chart saved to output/btc_trend.png")
