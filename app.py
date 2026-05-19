import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.title("Cryptocurrency Price Prediction")
symbol = st.selectbox("Select asset", ["BTC-USD", "ETH-USD", "SOL-USD"])

if st.button("Run Prediction"):
    with st.spinner("Downloading data and training model..."):
        data = yf.download(symbol, period="2y")
        prices = data['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)

        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        predicted = scaler.inverse_transform(model.predict(X_test))
        real = scaler.inverse_transform(y_test)

    st.success("Done!")
    st.line_chart({"Real Price": real.flatten(), "Predicted Price": predicted.flatten()})
    st.warning("This is for educational purposes only. Not financial advice.")