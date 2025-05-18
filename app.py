import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/stock_model.pkl")

# UI
st.title("📈 Stock Price Predictor")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

# Upload data or fetch from API
uploaded_file = st.file_uploader("Upload CSV with historical prices")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.tail())

    # Feature preparation
    # ... (depends on your model)

    # Predict
    prediction = model.predict(…)  # Format inputs appropriately
    st.write(f"📊 Predicted Next Price: ${prediction[-1]:.2f}")

    # Plotting
    plt.plot(data['Close'])
    st.pyplot(plt)
