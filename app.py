import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os

from sklearn.preprocessing import MinMaxScaler

# App title
st.title("📈 Stock Price Prediction using LSTM")

# Upload file
uploaded_file = st.file_uploader("Upload stock price CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    if 'Close/Last' not in df.columns:
        st.error("❌ CSV must contain a 'Close' column.")
    else:
        # Clean 'Close' column if it contains '$'
        df['Close'] = df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)

        # Load model and scaler
        model = tf.keras.models.load_model('stock_price.h5')
        scaler = joblib.load('scaler.pkl')  # Make sure you saved this from training

        # Preprocessing
        close_prices = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)

        # Prepare input for prediction
        n_steps = 60
        X_test = []
        y_test = []

        for i in range(n_steps, len(scaled_data)):
            X_test.append(scaled_data[i-n_steps:i])
            y_test.append(scaled_data[i])

        X_test, y_test = np.array(X_test), np.array(y_test)

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot predictions vs actual
        st.subheader("📊 Actual vs Predicted Stock Prices")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_actual, label='Actual Price', color='red')
        ax.plot(predictions, label='Predicted Price', color='green')
        ax.set_title("Actual vs Predicted Stock Prices")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Stock Price")
        ax.legend()
        st.pyplot(fig)
