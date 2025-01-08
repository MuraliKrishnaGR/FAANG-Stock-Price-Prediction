import streamlit as st
import pandas as pd
import mlflow.pyfunc
import numpy as np
import matplotlib.pyplot as plt

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered model
model_name = "rfr_model"
model_version = 9  # Change the version as needed
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# Streamlit app title
st.title("FAANG Stock Price Prediction")

# Add a brief description of the app
st.markdown("""
This application predicts the closing price of selected FAANG stocks based on user-provided input features such as Open Price, High Price, Low Price, Volume, and Day of Week.
""")

# User Input Section
st.header("Input Stock Details")

# Input fields for user to enter stock data
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=105.0)
low_price = st.number_input("Low Price", value=95.0)
volume = st.number_input("Volume", value=1_000_000.0)
day_of_week = st.number_input("Day of Week (0=Mon, ..., 6=Sun)", value=0, min_value=0, max_value=6)

# Ticker selection (One-hot encoding for simplicity)
ticker = st.selectbox("Select Stock Ticker", ["AMZN", "GOOGL", "NFLX"])
ticker_amzn = 1 if ticker == "AMZN" else 0
ticker_googl = 1 if ticker == "GOOGL" else 0
ticker_nflx = 1 if ticker == "NFLX" else 0

# Prepare input DataFrame for prediction
user_input = pd.DataFrame({
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price],
    'Volume': [volume],
    'Day_of_Week': [day_of_week],
    'Ticker_AMZN': [ticker_amzn],
    'Ticker_GOOGL': [ticker_googl],
    'Ticker_NFLX': [ticker_nflx]
})

# Display user input as a table
st.subheader("Input Data Overview")
st.dataframe(user_input)

# Prediction button
if st.button("Predict Closing Price"):
    try:
        # Make prediction
        prediction_log = model.predict(user_input)
        prediction = np.exp(prediction_log)  # Back-transform if the model output is in log scale
        
        # Display the predicted stock price
        st.header("Predicted Stock Price")
        st.write(f"The predicted stock closing price is: ${prediction[0]:.2f}")
        
        # Visualization: Plot the input features and predicted price
        st.subheader("Prediction Visualization")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot the numerical inputs as bars
        ax1.bar(['Open', 'High', 'Low', 'Volume'], 
                [open_price, high_price, low_price, volume], 
                color='skyblue', label='Input Features')
        ax1.set_ylabel('Values')
        ax1.set_xlabel('Features')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')

        # Overlay the predicted price as a line
        ax2 = ax1.twinx()
        ax2.plot(['Day_of_Week', 'Prediction'], [day_of_week, prediction[0]], 
                 color='orange', marker='o', label='Day of Week and Prediction')
        ax2.set_ylabel('Price in $')
        ax2.legend(loc='upper right')
        
        # Set plot title
        plt.title('Input Features and Predicted Stock Price')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred while predicting: {e}")
