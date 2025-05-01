import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import datetime

st.title('Stock Price Prediction App (Scikit-Learn Version)')

#Sidebar
st.sidebar.header('Settings')
stock_symbol = st.sidebar.text_input('Enter Stock Symbol (Example: AAPL)', 'AAPL')
start_date = st.sidebar.date_input('Start Date', datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.date.today())
prediction_days = st.sidebar.slider('Prediction Days Into Future', 7, 60, 30)

#Download Data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

data = load_data(stock_symbol, start_date, end_date)

if data.empty:
    st.error("No data found! Please check the stock symbol or date range.")
    st.stop()

#Raw Data
st.subheader('Raw Data')
st.write(data.tail())

#Moving Averages
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

st.subheader('Stock Price with Moving Averages')
fig1, ax1 = plt.subplots()
ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
ax1.plot(data.index, data['SMA_50'], label='50-Day SMA', color='orange')
ax1.plot(data.index, data['SMA_200'], label='200-Day SMA', color='green')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.set_title(f'{stock_symbol} Closing Price + Moving Averages')
ax1.legend()
st.pyplot(fig1)

#Feature Engineering for Prediction
data['Prediction'] = data['Close'].shift(-prediction_days)

X = np.array(data[['Close']])[:-prediction_days]
y = np.array(data['Prediction'])[:-prediction_days]

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#Model Evaluation
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

st.subheader('Model Performance')
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")

#Actual vs Predicted Plot
st.subheader('Actual vs Predicted Close Price')
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, predictions, color='purple')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax2.set_xlabel('Actual Close Price')
ax2.set_ylabel('Predicted Close Price')
ax2.set_title('Actual vs Predicted')
st.pyplot(fig2)

#Future Forecast
forecast_input = np.array(data[['Close']])[-prediction_days:]
forecast_prediction = model.predict(forecast_input)

future_dates = pd.date_range(end_date, periods=prediction_days+1).to_pydatetime().tolist()[1:]
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': forecast_prediction})

st.subheader(f'🔮 Next {prediction_days} Days Forecast')
st.write(forecast_df)

#Forecast Line Chart
st.subheader('Forecasted Closing Prices')
fig3, ax3 = plt.subplots()
ax3.plot(forecast_df['Date'], forecast_df['Predicted Close Price'], marker='o', linestyle='-')
ax3.set_xlabel('Date')
ax3.set_ylabel('Predicted Close Price')
ax3.set_title(f'Next {prediction_days} Days Forecast for {stock_symbol}')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

#Download Forecast CSV
st.download_button(
    label="Download Forecast as CSV",
    data=forecast_df.to_csv(index=False).encode('utf-8'),
    file_name=f'{stock_symbol}_forecast.csv',
    mime='text/csv',
)
