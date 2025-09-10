# Stock Market Visualizer

Streamlit app to visualize stock market data using yfinance and Plotly.

## Features
- Interactive candlestick charts with overlays: Moving Averages, RSI, Bollinger Bands.
- Portfolio upload (CSV/Excel) and approximate valuation.
- Correlation analysis for multiple tickers.
- Export charts as PNG or HTML.
- Save/load presets (session-level) and download presets JSON.

## Run locally
1. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate    # mac/linux
venv\Scripts\activate       # windows
