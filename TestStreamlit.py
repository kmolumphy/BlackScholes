import streamlit as st
import pandas as pd


def main():

    CreateStreamLitInterface()


def CreateStreamLitInterface():
    st.write("""
    # My first app
    Hello *world!*
    """)
    StockPrice = st.slider("Enter Stock Price:", 0, 100)

if __name__ == "__main__":
    main()