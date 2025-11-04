import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import math


# Define Black Scholes inputs as globals for ease of access
global_stock_price = 31.5
global_strike_price = 22.75
global_time_to_expiry = 3.5
global_volatility = 0.5
global_interest_rate = 0.05

# Define Heat Map Variables
global_map_dimension = 10
global_min_spot_price = 90 #global_stock_price * 0.9
global_max_spot_price = 110 #global_stock_price * 1.1
global_min_volatility = 0.01
global_max_volatility = 1

# Hold Graph Interval Values
global_volatility_intervals = []
global_spot_price_intervals = []

def main():
    CreateStreamLitInterface()
    #AppTest.run(*, timeout=None)


def CreateStreamLitInterface():
    st.write("Black Scholes")

    InitDefaultGlobals()

    # Create Input Var Widgets
    strikeprice = globals()['global_stock_price']
    global_stock_price = st.number_input("Stock Price", placeholder=strikeprice)
    global_strike_price = st.number_input("Strike Price", placeholder=globals()['global_strike_price'])
    global_time_to_expiry = st.number_input("Time to Expiration", placeholder=globals()['global_time_to_expiry'])
    global_volatility = st.number_input("Volatility", placeholder=globals()['global_volatility'])
    global_interest_rate = st.number_input("Risk-Free Interest Rate ", placeholder=globals()['global_interest_rate'])

    # Create heat maps based on inputs
    PutDF = pd.DataFrame(CreatePutData(), columns=[f'Col {i}' for i in range(global_map_dimension)])
    #PutDF.rename(index=global_volatility_intervals, inplace=True)
    PutDF.index = global_spot_price_intervals

    CallDF = pd.DataFrame(CreateCallData(), columns=[f'Col {i}' for i in range(global_map_dimension)])
    #CallDF.rename(columns=global_spot_price_intervals, inplace=True)
    CallDF.index = global_spot_price_intervals

    st.write("Put Heat Map")
    CreateHeatMap("PutHeatMap", PutDF)

    st.write("Call Heat Map")
    CreateHeatMap("Call Heat Map", CallDF)

def InitDefaultGlobals():
    #globals()['global_stock_price'] = 31.5
    #globals()['global_strike_price'] = 22.75
    #globals()['global_time_to_expiry'] = 3.5
    #globals()['global_volatility'] = 0.5
    #globals()['global_interest_rate'] = 0.05

    temp_global_volatility_intervals = []
    temp_global_spot_price_intervals = []
    for num in range(global_map_dimension):
        temp_global_volatility_intervals.append(GetVolatility(num))
        temp_global_spot_price_intervals.append(GetCurrentSpotPrice(num))

    globals()['global_volatility_intervals'] = temp_global_volatility_intervals
    globals()['global_spot_price_intervals'] = temp_global_spot_price_intervals    


def CreateHeatMap(title, dataframe):
    # Create a heatmap using Seaborn
    plt.figure(figsize=(global_map_dimension, global_map_dimension))
    sns.heatmap(dataframe, annot=True, fmt=".01f", cmap='coolwarm', square=True, xticklabels=True, yticklabels=True) #index="Volatility", columns="Spot Price", )

    # Show the plot in Streamlit
    st.pyplot(plt)

def CreateCallData():
    data = []
    for CurrentRow in range(global_map_dimension):
        row = []
        for CurrentColumn in range(global_map_dimension):
            row.append(CalculateCallOriginalBlackScholes(global_volatility_intervals[CurrentRow], global_spot_price_intervals[CurrentColumn]))
        data.append(row)    
    return data

def CreatePutData():
    data = []
    for CurrentRow in range(global_map_dimension):
        row = []
        for CurrentColumn in range(global_map_dimension):
            row.append(CalculatePutOriginalBlackScholes(global_volatility_intervals[CurrentRow], global_spot_price_intervals[CurrentColumn]))
        data.append(row)    
    return data

def CalculateCallOriginalBlackScholes(Volatility, SpotPrice):
    D1 = (math.log(SpotPrice / global_strike_price)) + ((global_interest_rate + (Volatility**2 / 2)) * global_time_to_expiry) / (Volatility * math.sqrt(global_time_to_expiry))
    D2 = D1 - Volatility * math.sqrt(global_time_to_expiry)
    CallPrice = (SpotPrice * norm.cdf(D1)) - (global_strike_price * math.exp(-global_interest_rate * global_time_to_expiry) * norm.cdf(D2))
    return CallPrice

def CalculatePutOriginalBlackScholes(Volatility, SpotPrice):
    D1 = (math.log(SpotPrice / global_strike_price)) + ((global_interest_rate + (Volatility**2 / 2)) * global_time_to_expiry) / (Volatility * math.sqrt(global_time_to_expiry))
    D2 = D1 - Volatility * math.sqrt(global_time_to_expiry)
    PutPrice = (global_strike_price * math.exp(-global_interest_rate * global_time_to_expiry) * norm.cdf(-D2)) - (SpotPrice * norm.cdf(-D1))
    return PutPrice

def GetVolatility(RowNum):
    Volatility = (RowNum - 0) * (global_max_volatility - global_min_volatility) / (global_map_dimension - 1) + global_min_volatility
    return round(Volatility, 2)

def GetCurrentSpotPrice(ColumnNum):
    SpotPrice = (ColumnNum - 0) * (global_max_spot_price - global_min_spot_price) / (global_map_dimension - 1) + global_min_spot_price
    return round(SpotPrice, 2)


if __name__ == "__main__":
    main()