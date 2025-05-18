import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# App title
st.title("📈 LSTM Stock Price Predictor")

# Load model
@st.cache_resource
def load_lstm_model():
    return load_model("models/stock_lstm_model.h5")  # Make sure this exists

model = load_lstm_model()

# Upload data
st.subheader("Upload Stock Price CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.tail())

    # Preprocessing (basic check)
    if 'Close' not in df.columns:
        st.error("CSV must contain a 'Close' column.")
    else:
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare input for prediction (last 60 days)
        sequence_length = 60
        if len(data) < sequence_length:
            st.warning("Not enough data. Need at least 60 data points.")
        else:
            input_seq = scaled_data[-sequence_length:]
            X_test = np.reshape(input_seq, (1, input_seq.shape[0], 1))

            prediction = model.predict(X_test)
            predicted_price = scaler.inverse_transform(prediction)[0][0]

            st.subheader("📊 Predicted Next Closing Price")
            st.success(f"${predicted_price:.2f}")

            # Plot the last 100 days + prediction
            fig, ax = plt.subplots()
            ax.plot(df['Close'][-100:], label='Historical Price')
            ax.plot(len(df)-1, predicted_price, 'ro', label='Predicted Next')
            ax.legend()
            st.pyplot(fig)

else:
    st.info("Upload a CSV file with a 'Close' column to get started.")
