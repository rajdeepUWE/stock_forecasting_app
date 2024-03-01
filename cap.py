import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import requests
import tempfile
import os
from tensorflow.keras.models import load_model

# Function to load Keras model from a URL
def load_keras_model_from_github(model_url):
    response = requests.get(model_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_model_file:
        temp_model_file.write(response.content)
        temp_model_file_path = temp_model_file.name
    keras_model = load_model(temp_model_file_path)
    os.unlink(temp_model_file_path)
    return keras_model

# Function to train SVR model
def train_svr_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    svr_model = SVR(kernel='rbf')
    svr_model.fit(scaler.transform(X), y)
    return svr_model, scaler

# Function to train Linear Regression model
def train_linear_regression_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    lr_model = LinearRegression()
    lr_model.fit(scaler.transform(X), y)
    return lr_model, scaler

# Function to train Random Forest model
def train_random_forest_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(scaler.transform(X), y)
    return rf_model, scaler

# Streamlit UI
def main():
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

    # Load the Keras model from GitHub
    model_url = 'https://github.com/rajdeepUWE/stock_forecasting_app/raw/master/model2.h5'
    keras_model = load_keras_model_from_github(model_url)
    st.success("Keras Neural Network model loaded successfully!")

    # Train SVR model
    svr_model, svr_scaler = train_svr_model(data)
    st.success("SVR model trained successfully!")

    # Train Linear Regression model
    lr_model, lr_scaler = train_linear_regression_model(data)
    st.success("Linear Regression model trained successfully!")

    # Train Random Forest model
    rf_model, rf_scaler = train_random_forest_model(data)
    st.success("Random Forest model trained successfully!")

    # Machine Learning Model Selection
    selected_model = st.selectbox('Select Model', ['Keras Neural Network', 'Support Vector Regressor (SVR)', 'Linear Regression', 'Random Forest'])

    # Model Training and Prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    y_true = data['Close'].values[-7:]  # Take the last 7 days' true values
    X_pred = np.arange(len(data), len(data) + 7).reshape(-1, 1)
    # Scale the input data for prediction
    X_pred_scaled = scaler.transform(X_pred)

    if selected_model == 'Keras Neural Network':
        model = keras_model
        y_pred = model.predict(X_pred_scaled).flatten()
    elif selected_model == 'Support Vector Regressor (SVR)':
        model = svr_model
        y_pred = model.predict(X_pred_scaled)
    elif selected_model == 'Linear Regression':
        model = lr_model
        y_pred = model.predict(X_pred_scaled)
    elif selected_model == 'Random Forest':
        model = rf_model
        y_pred = model.predict(X_pred_scaled)

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

if __name__ == "__main__":
    main()
