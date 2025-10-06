import pandas as pd
import yfinance as yf

try:
    data = yf.download("SPY", start="2015-01-01", end="2025-10-06")
except:
    data = pd.read_csv("data/SPY.csv", index_col=0, parse_dates=True)

print(data.head())
