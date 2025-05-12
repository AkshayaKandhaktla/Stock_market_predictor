import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS for red background
st.markdown(
    """
    <style>
    .stApp {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 SNIST App for Stock Price Predictor ")

# Multiselect widget for choosing multiple stocks
stocks = st.multiselect("Select Stock IDs", ["GOOG", "AAPL", "MSFT", "AMZN"], default=["GOOG"])

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Dictionary to store stock data
stock_data = {}

# Download stock data for each selected stock
for stock in stocks:
    stock_data[stock] = yf.download(stock, start, end)

# Calculate moving averages for each selected stock
for stock in stocks:
    stock_data[stock]['MA_for_250_days'] = stock_data[stock]['Close'].rolling(window=250).mean()
    stock_data[stock]['MA_for_200_days'] = stock_data[stock]['Close'].rolling(window=200).mean()
    stock_data[stock]['MA_for_100_days'] = stock_data[stock]['Close'].rolling(window=100).mean()

# Load the model without the optimizer
model = load_model(r"E:/Mr JP/2025/mini/stock market/stock_price_prediction-main/stock_price_prediction-main/Latest_stock_price_model.h5", compile=False)
# Compile the model with a compatible optimizer
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Display data and charts for each selected stock
for stock in stocks:
    st.subheader(f"{stock} Stock Data Overview")
    st.dataframe(stock_data[stock].describe())

    # Visualization function
    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, title=""):
        fig = plt.figure(figsize=figsize)
        plt.plot(values, 'orange', label='Moving Average')
        plt.plot(full_data.Close, 'b', label='Close Price')
        if extra_data:
            plt.plot(extra_dataset, 'g', label='Additional Moving Average')
        plt.title(title)
        plt.legend()
        return fig

    st.subheader(f"{stock} Moving Averages")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_graph((10, 6), stock_data[stock]['MA_for_250_days'], stock_data[stock], 0, title='MA for 250 days'))
        st.pyplot(plot_graph((10, 6), stock_data[stock]['MA_for_200_days'], stock_data[stock], 0, title='MA for 200 days'))
    with col2:
        st.pyplot(plot_graph((10, 6), stock_data[stock]['MA_for_100_days'], stock_data[stock], 0, title='MA for 100 days'))
        st.pyplot(plot_graph((10, 6), stock_data[stock]['MA_for_100_days'], stock_data[stock], 1, stock_data[stock]['MA_for_250_days'], title='MA for 100 and 250 days'))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[stock][['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    predictions = model.predict(x_data)

    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    ploting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index=stock_data[stock].index[len(stock_data[stock]) - len(inv_pre):]
    )

    st.subheader(f"{stock} Original values vs Predicted values")
    st.line_chart(ploting_data)

    st.subheader(f"{stock} Close Price vs Predicted Close Price")
    fig = plt.figure(figsize=(15, 6))
    plt.plot(stock_data[stock].Close[:len(stock_data[stock]) - len(ploting_data)], 'b', label='Data - not used')
    plt.plot(ploting_data.index, ploting_data['original_test_data'], 'orange', label='Original Test Data')
    plt.plot(ploting_data.index, ploting_data['predictions'], 'green', label='Predicted Test Data')
    plt.legend()
    plt.title(f"{stock} Original Close Price vs Predicted Close Price")
    st.pyplot(fig)
