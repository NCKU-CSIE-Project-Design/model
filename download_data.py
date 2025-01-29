import yfinance as yf
import pandas as pd
import mplfinance as mpf


symbol = "^GSPC" # S&P500
start_date = "2025-01-01"
end_date = "2025-01-27"

data = yf.download(symbol, start=start_date, end=end_date, interval="1d")

if data.empty:
    print("failed")
else:
    print("Success")

# Flatten the MultiIndex columns
data.columns = data.columns.droplevel(1)  # Remove the '^GSPC' level
print(data.columns)

# Save data
data.to_csv("SPX500.csv")

# Create the candlestick chart
mpf.plot(data, 
         type='candle',
         title=f'{symbol} Stock Price',
         volume=True,
         style='yahoo',
         figsize=(12, 8),
         savefig='sp500_chart.png')

