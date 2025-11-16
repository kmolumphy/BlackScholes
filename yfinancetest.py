import yfinance as yf
from datetime import datetime

company_name = "temp"

ticker = yf.Ticker(company_name)
info = ticker.info

print(ticker.info.get('currentPrice'))
print(ticker.info['regularMarketPrice'])

current_datetime = datetime.today()
year_ago_datetime = datetime(year=(current_datetime.year-1), month=current_datetime.month, day=current_datetime.day)

# Fetch historical data
stock_data = yf.download(company_name, start=year_ago_datetime, end=current_datetime)
#print(stock_data)
# Calculate daily returns (percentage change)
daily_returns = stock_data['Close'].pct_change().dropna()

#Calculate daily volatility (standard deviation of daily returns)
daily_volatility = daily_returns.std()
