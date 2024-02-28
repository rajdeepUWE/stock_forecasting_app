import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Load the pre-trained Keras model
keras_model = load_model('Stock Predictions Model1.keras')

# Function to train and predict using Linear Regression
def linear_regression_predict(train_data, test_data):
    lr_model = LinearRegression()
    lr_model.fit(train_data, np.ravel(train_data))
    predictions = lr_model.predict(test_data)
    return predictions

# Function to train and predict using Random Forest Regressor
def random_forest_predict(train_data, test_data):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_data, np.ravel(train_data))
    predictions = rf_model.predict(test_data)
    return predictions

# Function to train and predict using Support Vector Regressor
def svr_predict(train_data, test_data):
    svr_model = SVR(kernel='rbf', C=1000, gamma=0.1)
    svr_model.fit(train_data, np.ravel(train_data))
    predictions = svr_model.predict(test_data)
    return predictions

# Function to calculate moving average
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Function to forecast next 7 days' stock prices
def forecast_next_7_days(data, model, scaler):
    last_100_days = data[-100:].values.reshape(-1, 1)  # Reshape to 2D array with 1 column
    scaled_last_100_days = scaler.transform(last_100_days)
    x_pred = scaled_last_100_days.reshape(1, 100, 1)
    forecasts = []
    for _ in range(7):
        next_day_pred = model.predict(x_pred)[0, 0]
        forecasts.append(next_day_pred)
        x_pred = np.roll(x_pred, -1)
        x_pred[0, -1, 0] = next_day_pred
    forecasts = np.array(forecasts).reshape(-1, 1)
    return scaler.inverse_transform(forecasts).flatten()

# Function to calculate evaluation metrics
def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mse, mae, r2

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
ma_100_days = calculate_moving_average(data['Close'], 100)
ma_200_days = calculate_moving_average(data['Close'], 200)

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
    'Linear Regression': linear_regression_predict,
    'Random Forest Regressor': random_forest_predict,
    'Support Vector Regressor': svr_predict
}

selected_model = st.selectbox('Select Model', list(ml_models.keys()))

# Model Training and Prediction
if selected_model == 'Keras Neural Network':
    model = ml_models[selected_model]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    last_100_days = data['Close'].values[-100:].reshape(-1, 1)
    scaled_last_100_days = scaler.transform(last_100_days)
    y_true = data['Close'].values[-7:]  # Take the last 7 days' true values
    y_pred = forecast_next_7_days(data['Close'], model, scaler)
    y_pred = y_pred[-7:]  # Take the last 7 days' predicted values

elif selected_model in ['Linear Regression', 'Random Forest Regressor', 'Support Vector Regressor']:
    model = ml_models[selected_model]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    x_train = data['Close'].values[:-7].reshape(-1, 1)
    y_train = data['Close'].values[7:]
    x_test = data['Close'].values[-7:].reshape(-1, 1)
    y_true = data['Close'].values[-7:]
    y_pred = model(x_train, x_test)

# Evaluation Metrics
rmse, mse, mae, r2 = evaluate_predictions(y_true, y_pred)

st.subheader('Evaluation Metrics')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')
st.write(f'Mean Squared Error (MSE): {mse}')
st.write(f'Mean Absolute Error (MAE): {mae}')
st.write(f'R-squared (R2) Score: {r2}')

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
