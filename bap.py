import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import tempfile
import os

# Initialize keras_model variable to None
keras_model = None

# URL of the raw Keras model file on GitHub
model_url = 'https://github.com/rajdeepUWE/stock_forecasting_app/raw/master/model2.h5'

# Fetch the model file from GitHub
response = requests.get(model_url)
response.raise_for_status()  # Raise an exception for any HTTP error

try:
    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_model_file:
        temp_model_file.write(response.content)
        temp_model_file_path = temp_model_file.name

    # Load the Keras model from the temporary file
    keras_model = load_model(temp_model_file_path)
    st.success("Model loaded successfully!")
finally:
    # Delete the temporary file
    os.unlink(temp_model_file_path)

# Function to forecast next 7 days' stock prices using Keras model
def forecast_next_7_days_keras(data, model, scaler):
    last_100_days = data[-100:].values.reshape(-1, 1)
    scaled_last_100_days = scaler.transform(last_100_days)
    x_pred = scaled_last_100_days[-7:].reshape(1, 7, 1)  # Adjusted to predict 7 days
    forecasts = []
    for _ in range(7):
        next_day_pred = model.predict(x_pred)[0, 0]
        forecasts.append(next_day_pred)
        x_pred = np.roll(x_pred, -1, axis=1)
        x_pred[0, -1, 0] = next_day_pred
    forecasts = np.array(forecasts).reshape(-1, 1)
    return scaler.inverse_transform(forecasts).flatten()

# Streamlit UI
st.title('Stock Market Predictor')

# Sidebar: Input parameters
st.sidebar.subheader('Input Parameters')
stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
start_date = st.sidebar.date_input('Select Start Date', pd.to_datetime('1985-01-01'))
end_date = st.sidebar.date_input('Select End Date', pd.to_datetime('today'))

# Fetch stock data
data = yf.download(stock, start=start_date, end=end_date)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Calculate moving averages
ma_100_days = data['Close'].rolling(window=100).mean()
ma_200_days = data['Close'].rolling(window=200).mean()

# Plot moving averages
st.subheader('Moving Average Plots')
fig_ma100 = go.Figure()
fig_ma100.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100'))
fig_ma100.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
fig_ma100.update_layout(title='Price vs MA100', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma100)

fig_ma200 = go.Figure()
fig_ma200.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
fig_ma200.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
fig_ma200.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='green')))
fig_ma200.update_layout(title='Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma200)

# Machine Learning Model Selection
ml_models = {
    'Keras Neural Network': keras_model,
}

selected_model = st.selectbox('Select Model', list(ml_models.keys()))

# Model Training and Prediction
if selected_model == 'Keras Neural Network' and keras_model is not None:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    y_true = data['Close'].values[-7:]  # Take the last 7 days' true values
    y_pred = forecast_next_7_days_keras(data['Close'], keras_model, scaler)

    # Plot Original vs Predicted Prices
    st.subheader('Original vs Predicted Prices')
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=data.index[-7:], y=y_pred, mode='lines', name='Predicted Price',
                                   hovertemplate='Date: %{x}<br>Predicted Price: %{y:.2f}<extra></extra>'))
    fig_pred.add_trace(go.Scatter(x=data.index[-7:], y=y_true, mode='lines', name='Original Price',
                                   hovertemplate='Date: %{x}<br>Original Price: %{y:.2f}<extra></extra>'))
    fig_pred.update_layout(title='Original Price vs Predicted Price', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_pred)

    # Forecasted Prices for Next 7 Days
    st.subheader('Next 7 Days Forecasted Close Prices')
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Close Price': y_pred})
    st.write(forecast_df)
