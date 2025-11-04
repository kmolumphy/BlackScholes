import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Define Black Scholes inputs as globals for ease of access
global_stock_price = 100
global_strike_price = 110
global_time_to_expiry = 1.1
global_volatility = 0.2
global_interest_rate = 0.1

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
    global_stock_price = st.number_input("Stock Price", placeholder=globals()['global_strike_price'])
    global_strike_price = st.number_input("Strike Price", placeholder=globals()['global_strike_price'])
    global_time_to_expiry = st.number_input("Time to Expiration", placeholder=globals()['global_time_to_expiry'])
    global_volatility = st.number_input("Volatility", placeholder=globals()['global_volatility'])
    global_interest_rate = st.number_input("Risk-Free Interest Rate ", placeholder=globals()['global_interest_rate'])

    # Create heat maps based on inputs
    PutDF = pd.DataFrame(CreatePutData(), columns=[f'Col {i}' for i in range(global_map_dimension)])
    #PutDF.columns = global_volatility_intervals
    CallDF = pd.DataFrame(CreateCallData(), columns=[f'Col {i}' for i in range(global_map_dimension)])
    CreateHeatMap("PutHeatMap", PutDF)
    CreateHeatMap("Call Heat Map", CallDF)

def InitDefaultGlobals():
    globals()['global_stock_price'] = 100
    globals()['global_strike_price'] = 110
    globals()['global_time_to_expiry'] = 1.1
    globals()['global_volatility'] = 0.2
    globals()['global_interest_rate'] = 0.1

    globals()['global_interest_rate'] = 10

    temp_global_volatility_intervals = []
    temp_global_spot_price_intervals = []
    for num in range(global_map_dimension):
        temp_global_volatility_intervals.append(GetVolatility(num))
        temp_global_spot_price_intervals.append(GetCurrentSpotPrice(num))

    globals()['global_volatility_intervals'] = temp_global_volatility_intervals
    globals()['global_spot_price_intervals'] = temp_global_spot_price_intervals
    

def CreatePutHeatMap():
    fig, ax = plt.subplots() 

def CreateHeatMap(title, dataframe):
    # Create a heatmap using Seaborn
    plt.figure(figsize=(global_map_dimension, global_map_dimension),)
    sns.heatmap(dataframe, annot=True, cmap='coolwarm', square=True) #index="Volatility", columns="Spot Price", )

    # Show the plot in Streamlit
    st.pyplot(plt)


def CreateCallData():
    data = []
    for CurrentRow in range(global_map_dimension):
        row = []
        for CurrentColumn in range(global_map_dimension):
            row.append(CalculateCall(global_volatility_intervals[CurrentRow], global_spot_price_intervals[CurrentColumn]))
        data.append(row)    
    return data



def CreatePutData():
    data = []
    for CurrentRow in range(global_map_dimension):
        row = []
        for CurrentColumn in range(global_map_dimension):
            row.append(CalculatePut(global_volatility_intervals[CurrentRow], global_spot_price_intervals[CurrentColumn]))
        data.append(row)    
    return data

def CalculateCall(Volatility, SpotPrice):
    # Temp return val to test
    return Volatility * SpotPrice

def CalculatePut(Volatility, SpotPrice):
    # Temp return val to test
    return Volatility * SpotPrice

def GetVolatility(RowNum):
    CurrentVolatility = (RowNum / (global_map_dimension - 1)) * ((global_max_volatility - global_min_volatility) + global_min_volatility)
    return CurrentVolatility

def GetCurrentSpotPrice(ColumnNum):
    CurrentVolatility = (ColumnNum / (global_map_dimension - 1)) * ((global_max_spot_price - global_min_spot_price) + global_min_spot_price)
    return CurrentVolatility


if __name__ == "__main__":
    main()