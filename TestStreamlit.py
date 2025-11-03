import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Define Black Scholes inputs as globals for ease of access
global_stock_price = 100
global_strike_price = 110
global_time_to_expiry = 0.1
global_volatility = 0.2
global_interest_rate = 110


def main():
    CreateStreamLitInterface()
    AppTest.run(*, timeout=None)


def CreateStreamLitInterface():
    st.write("Black Scholes")

    global_stock_price = st.number_input("Stock Price")
    global_strike_price = st.number_input("Strike Price")
    global_time_to_expiry = st.number_input("Time to Expiration")
    global_volatility = st.number_input("Volatility")
    global_interest_rate = st.number_input("Risk-Free Interest Rate ")


def CreateStockHeatMap():
    fig, ax = plt.subplots()  ## <- Create matplotlib Figure & Axes

    july.heatmap(
        dates=df.Dates,
        data=df.Pages,
        cmap='github',
        month_grid=True,
        horizontal=True,
        value_label=True,
        date_label=False,
        weekday_label=True,
        month_label=True,
        year_label=True,
        colorbar=True,
        fontfamily="monospace",
        fontsize=12,
        title="Daily Pages Read",
        ax=ax   ## <- Tell July to put the heatmap in that Axes
    )

    st.pyploy(fig)

if __name__ == "__main__":
    main()