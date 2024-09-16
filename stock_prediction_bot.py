import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from ta import add_all_ta_features
import matplotlib.pyplot as plt

# Step 1: Get Historical Stock Data
@st.cache_data
def get_stock_data(ticker):
    data = yf.download(ticker, period='1y', interval='1wk')
    return data

# Step 2: Preprocess Data and Add Technical Indicators
def preprocess_data(df):
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    # Label the target as 1 (Bullish) or 0 (Bearish) for next week's close
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df = df.dropna()  # Remove any NaN values
    return df

# Step 3: Train the Model (Random Forest)
def train_model(df):
    X = df.drop(['Target'], axis=1)  # Features
    y = df['Target']  # Target (Bullish or Bearish)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model

# Step 4: Make Prediction
def predict_next_week(model, df):
    last_week_data = df.tail(1).drop(['Target'], axis=1)
    prediction = model.predict(last_week_data)
    
    if prediction[0]:
        return "Bullish (Buy)"
    else:
        return "Bearish (Sell)"

# Step 5: Plot Stock Data
def plot_stock(df, ticker):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Step 6: Streamlit Web App
def main():
    st.title("AI Stock Prediction Bot")
    ticker = st.text_input("Enter stock ticker (e.g. AAPL):")

    if ticker:
        with st.spinner('Fetching stock data...'):
            df = get_stock_data(ticker)
        
        st.write(f"Displaying last 1 year data for {ticker}:")
        st.dataframe(df)
        
        df = preprocess_data(df)
        model = train_model(df)
        prediction = predict_next_week(model, df)
        
        st.write(f"Prediction for {ticker}: **{prediction}**")
        
        st.write("Here is the stock chart:")
        plot_stock(df, ticker)

if __name__ == '__main__':
    main()
