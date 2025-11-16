import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import math
import yfinance as yf
from datetime import datetime


# Define Black Scholes inputs as globals for ease of access
#global_stock_price = 31.5
#global_strike_price = 22.75
#global_time_to_expiry = 3.5
#global_volatility = 0.5
#global_interest_rate = 0.05

# Define Heat Map Variables
global_map_dimension = 11 # Default to 11 so input vars have output for exact val in center

# Hold Graph Interval Values
global_volatility_intervals = []
global_spot_price_intervals = []

def main():
    CreateStreamLitInterface()
    #AppTest.run(*, timeout=None)


def CreateStreamLitInterface():
    st.title("Black Scholes")
    st.set_page_config(layout="wide")

    with st.sidebar:
        st.write("Input Parameters:")

        default_stock_price = 30.00
        default_volatility = 0.50
        default_strike_price = 20.00
        default_time_to_expiry = 1.00
        default_global_interest_rate = 0.05

        # Check population of ticker symbol input
        company_symbol = st.text_input("Company Symbol", value="none",  placeholder="Enter Company Symbol: ").upper()
        if (company_symbol != ""):
            company_ticker = yf.Ticker(company_symbol)
            # Fetch historical data, we eventually want to store this info to avoid repeated downloads
            current_datetime = datetime.today()
            year_ago_datetime = datetime(year=(current_datetime.year-1), month=current_datetime.month, day=current_datetime.day)
            stock_data = yf.download(company_symbol, start=year_ago_datetime, end=current_datetime)

            if (len(company_ticker.info) > 1 and stock_data.size != 0): # This means company info is valid
                default_stock_price = GetStockPrice(company_ticker)
                calculated_volatility = CalculateVolatility(stock_data, company_symbol)
                if (calculated_volatility != 0): default_volatility = calculated_volatility
                st.write("Valid Symbol Found! Finding Stock Price and Calculating Volatility (1 yr)")
            elif (company_symbol != "none"):
                st.write("No Valid Symbol Found!")

        global_stock_price = st.number_input("Stock Price", value=default_stock_price, min_value=0.01, format="%.2f", placeholder="Enter Stock Price: ")
        global_volatility = st.number_input("Volatility", value=default_volatility, min_value=0.01, format="%.5f", placeholder="Enter Volatility: ")
        global_strike_price = st.number_input("Strike Price", value=default_strike_price, min_value=0.01, format="%.2f", placeholder="Enter Strike Price: ")
        global_time_to_expiry = st.number_input("Time to Expiration", value=default_time_to_expiry, min_value=0.01, format="%.2f", placeholder="Enter Time To Expiry: (yrs)")
        global_interest_rate = st.number_input("Risk-Free Interest Rate ", value=default_global_interest_rate, min_value=0.01, format="%.2f", placeholder="Enter Global Interest Rate: ")

        global_volatility_variability = st.slider("Volatility Variability", value=0.5, format="%.2f")
        global_spot_price_variability = st.slider("Spot Price Variability", value=0.5, format="%.2f")

    InitDefaultGlobals(global_volatility, global_stock_price, global_volatility_variability, global_spot_price_variability)

    #global_strike_price = max(0.01, global_strike_price) # Cannot be 0

    # Create heat maps based on inputs
    PutData = CreatePutData(global_strike_price, global_time_to_expiry, global_interest_rate)
    PutDF = pd.DataFrame(PutData, columns=[global_volatility_intervals[i] for i in range(global_map_dimension)])

    PutDF.index = global_spot_price_intervals #Edit row labels

    CallData = CreateCallData(global_strike_price, global_time_to_expiry, global_interest_rate)
    CallDF = pd.DataFrame(CallData, columns=[global_volatility_intervals[i] for i in range(global_map_dimension)])

    CallDF.index = global_spot_price_intervals  #Edit row labels
    
    # Create two columns
    col1, col2 = st.columns(2,gap="small",vertical_alignment="center",)

    # Place the first chart in the first column
    with col1:
        CreateHeatMap("Put Heat Map", PutDF)

    # Place the second chart in the second column
    with col2:
        CreateHeatMap("Call Heat Map", CallDF)


def CalculateVolatility(stock_data ,company_name):
    # Calculate daily returns (percentage change)
    daily_returns = stock_data['Close'].pct_change().dropna()

    # Calculate daily volatility (standard deviation of daily returns)
    daily_volatility = daily_returns.std()
    return daily_volatility[company_name]
    
def GetStockPrice(company_ticker):
    return company_ticker.info.get('regularMarketPrice')

def InitDefaultGlobals(volatility, spot_price, volatility_variability, spot_price_variability):
    temp_global_volatility_intervals = []
    temp_global_spot_price_intervals = []
    global_min_volatility = volatility + volatility * -volatility_variability
    global_max_volatility = volatility + volatility * volatility_variability
    global_min_spot_price = spot_price + spot_price * -spot_price_variability
    global_max_spot_price = spot_price + spot_price * spot_price_variability
    for num in range(global_map_dimension):
        temp_global_volatility_intervals.append(GetVolatility(num, global_min_volatility, global_max_volatility))
        temp_global_spot_price_intervals.append(GetCurrentSpotPrice(num, global_min_spot_price, global_max_spot_price))

    globals()['global_volatility_intervals'] = temp_global_volatility_intervals
    globals()['global_spot_price_intervals'] = temp_global_spot_price_intervals    


def CreateHeatMap(title, dataframe):
    # Create a heatmap using Seaborn
    plt.figure(figsize=(global_map_dimension, global_map_dimension))
    sns.heatmap(dataframe, annot=True, fmt=".2f", cbar=False, cmap='RdYlGn', square=True, xticklabels=True, yticklabels=True)
    plt.title(title, fontsize = 20)
    plt.xlabel('Volatility', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Spot Price', fontsize = 15) # y-axis label with fontsize 15
    # Show the plot in Streamlit
    st.pyplot(plt)

def CreateCallData(strike_price, time_to_expiry, interest_rate):
    data = []
    for CurrentRow in range(global_map_dimension):
        row = []
        for CurrentColumn in range(global_map_dimension):
            row.append(CalculateCallOriginalBlackScholes(global_volatility_intervals[CurrentRow], global_spot_price_intervals[CurrentColumn], strike_price, time_to_expiry, interest_rate))
        data.append(row)    
    return data

def CreatePutData(strike_price, time_to_expiry, interest_rate):
    data = []
    for CurrentRow in range(global_map_dimension):
        row = []
        for CurrentColumn in range(global_map_dimension):
            row.append(CalculatePutOriginalBlackScholes(global_volatility_intervals[CurrentRow], global_spot_price_intervals[CurrentColumn], strike_price, time_to_expiry, interest_rate))
        data.append(row)    
    return data

def CalculateCallOriginalBlackScholes(volatility_interval, spot_price_interval, strike_price, time_to_expiry, interest_rate):
    D1 = (math.log(spot_price_interval / strike_price)) + ((interest_rate + (volatility_interval**2 / 2)) * time_to_expiry) / (volatility_interval * math.sqrt(time_to_expiry))
    D2 = D1 - volatility_interval * math.sqrt(time_to_expiry)
    CallPrice = (spot_price_interval * norm.cdf(D1)) - (strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(D2))
    return CallPrice

def CalculatePutOriginalBlackScholes(volatility_interval, spot_price_interval, strike_price, time_to_expiry, interest_rate):
    D1 = (math.log(spot_price_interval / strike_price)) + ((interest_rate + (volatility_interval**2 / 2)) * time_to_expiry) / (volatility_interval * math.sqrt(time_to_expiry))
    D2 = D1 - volatility_interval * math.sqrt(time_to_expiry)
    PutPrice = (strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(-D2)) - (spot_price_interval * norm.cdf(-D1))
    return PutPrice

def GetVolatility(RowNum, global_min_volatility, global_max_volatility):
    Volatility = (RowNum - 0) * (global_max_volatility - global_min_volatility) / (global_map_dimension - 1) + global_min_volatility
    return round(Volatility, 2)

def GetCurrentSpotPrice(ColumnNum, global_min_spot_price, global_max_spot_price):
    SpotPrice = (ColumnNum - 0) * (global_max_spot_price - global_min_spot_price) / (global_map_dimension - 1) + global_min_spot_price
    return round(SpotPrice, 2)


if __name__ == "__main__":
    main()