import streamlit as st
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from textblob import TextBlob
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from functools import lru_cache
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Popular stocks dictionary
POPULAR_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX",
    "Alphabet": "GOOGL",
    "Adobe": "ADBE"
}

class StockSentimentPredictor:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.cache = {}
        self.model = None
        self.scaler = StandardScaler()
        
    @lru_cache(maxsize=128)
    def get_stock_data(self, ticker, period='6mo'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if len(df) < 30:  # Ensure we have enough data
                st.warning(f"Not enough data for {ticker}")
                return pd.DataFrame()
            return df
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()

    @lru_cache(maxsize=128)
    def get_news_sentiment(self, query, max_articles=3):
        """Get news articles and calculate sentiment using TextBlob"""
        try:
            articles = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=max_articles
            )['articles']
            
            sentiments = []
            for article in articles:
                text = article['title'] + ' ' + article['description']
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                
                # Categorize sentiment
                if sentiment > 0.2:
                    category = 'Positive'
                elif sentiment < -0.2:
                    category = 'Negative'
                else:
                    category = 'Neutral'
                
                sentiments.append({
                    'sentiment': sentiment,
                    'category': category,
                    'title': article['title']
                })
            
            return {
                'average': np.mean([s['sentiment'] for s in sentiments]),
                'articles': sentiments
            } if sentiments else {'average': 0, 'articles': []}
        except Exception as e:
            st.warning(f"News API Error: {str(e)}")
            return {'average': 0, 'articles': []}

    def calculate_daily_change(self, data):
        """Calculate daily price change percentage"""
        if len(data) > 1:
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2]
            return ((current_price - previous_price) / previous_price) * 100
        return 0

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        try:
            # Calculate moving averages
            data['MA5'] = data['Close'].rolling(window=5, min_periods=1).mean()
            data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Fill NaN values with 0 for the first few rows
            data = data.fillna(0)
            
            return data
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return data

    def prepare_features(self, data, sentiment):
        """Prepare features for the prediction model"""
        try:
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(data)
            
            # Create features
            features = pd.DataFrame({
                'Close': indicators['Close'],
                'Volume': indicators['Volume'],
                'MA5': indicators['MA5'],
                'MA20': indicators['MA20'],
                'RSI': indicators['RSI'],
                'MACD': indicators['MACD'],
                'Signal': indicators['Signal'],
                'Sentiment': sentiment,
                'Volatility': indicators['Close'].pct_change().rolling(window=20).std(),
                'Price_Change': indicators['Close'].pct_change(),
                'Volume_Change': indicators['Volume'].pct_change()
            })
            
            # Drop rows with NaN values
            features = features.dropna()
            
            # Create target variable
            features['Target'] = features['Close'].shift(-1)
            features = features.dropna()
            
            return features
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def train_model(self, features):
        """Train the prediction model"""
        if len(features) < 30:  # Return 0 error if not enough data
            return 0
            
        X = features.drop(['Target', 'Close'], axis=1)
        y = features['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with optimized parameters
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Calculate directional accuracy
        # Convert to numpy arrays for direct comparison
        y_test_np = np.array(y_test)
        y_pred_np = np.array(y_pred)
        
        # Calculate differences
        actual_diff = np.diff(y_test_np)
        pred_diff = np.diff(y_pred_np)
        
        # Get direction (1 for positive, -1 for negative)
        actual_direction = np.sign(actual_diff)
        predicted_direction = np.sign(pred_diff)
        
        # Calculate accuracy
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }

    def predict_price(self, features):
        """Predict future price"""
        if self.model is None:
            return None
            
        # Prepare latest data
        latest_data = features.iloc[-1].drop(['Target', 'Close']).values.reshape(1, -1)
        
        # Scale the data
        latest_data_scaled = self.scaler.transform(latest_data)
        
        # Make prediction
        prediction = self.model.predict(latest_data_scaled)[0]
        return prediction

    def create_dashboard(self):
        """Create the Streamlit dashboard"""
        st.title("Stock Analysis Dashboard")
        
        # Create a dropdown menu for stock selection
        selected_stock = st.selectbox(
            "Select a Stock",
            options=list(POPULAR_STOCKS.keys()),
            format_func=lambda x: x
        )
        
        # Get the ticker symbol
        ticker = POPULAR_STOCKS[selected_stock]
        
        if ticker:
            try:
                with st.spinner('Loading data and training model...'):
                    # Get stock data
                    stock_data = self.get_stock_data(ticker, period='6mo')
                    
                    if stock_data.empty:
                        st.error("No data found for this ticker")
                        return
                    
                    # Get sentiment scores
                    sentiment_data = self.get_news_sentiment(ticker)
                    
                    # Calculate daily change
                    daily_change = self.calculate_daily_change(stock_data)
                    
                    # Prepare features
                    features = self.prepare_features(stock_data, sentiment_data['average'])
                    
                    # Train model
                    metrics = self.train_model(features)
                    
                    # Make prediction
                    prediction = self.predict_price(features)
                    
                    # Create main price chart
                    fig = go.Figure()
                    
                    # Add stock price
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        name='Stock Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add moving averages
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=features['MA5'],
                        name='5-day MA',
                        line=dict(dash='dash', color='orange')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=features['MA20'],
                        name='20-day MA',
                        line=dict(dash='dash', color='green')
                    ))
                    
                    # Add volume as a secondary axis
                    fig.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name='Volume',
                        yaxis='y2',
                        marker_color='gray'
                    ))
                    
                    # Add RSI as a third axis
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=features['RSI'],
                        name='RSI',
                        yaxis='y3',
                        line=dict(color='purple')
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_stock} ({ticker}) Stock Analysis",
                        xaxis_title='Date',
                        yaxis_title='Price',
                        yaxis2=dict(
                            title='Volume',
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        yaxis3=dict(
                            title='RSI',
                            overlaying='y',
                            side='right',
                            range=[0, 100],
                            showgrid=False
                        ),
                        legend_title='Legend',
                        height=600,
                        margin=dict(l=50, r=100, t=50, b=50),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display metrics
                    st.subheader("Stock Analysis")
                    col1, col2 = st.columns(2)
                    col1.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
                    col2.metric("Daily Change", f"{daily_change:.2f}%")
                    
                    # Display prediction
                    st.subheader("Price Prediction")
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Price", f"${prediction:.2f}")
                    col2.metric("Prediction Error", f"{metrics['mse']:.2f}")
                    
                    # Display technical indicators
                    st.subheader("Technical Indicators")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RSI", f"{features['RSI'].iloc[-1]:.2f}")
                    col2.metric("MACD", f"{features['MACD'].iloc[-1]:.2f}")
                    col3.metric("Signal Line", f"{features['Signal'].iloc[-1]:.2f}")
                    
                    # Display sentiment analysis
                    st.subheader("Sentiment Analysis")
                    st.metric("News Sentiment", f"{sentiment_data['average']:.2f}")
                    
                    # Create sentiment distribution chart
                    sentiment_df = pd.DataFrame(sentiment_data['articles'])
                    if not sentiment_df.empty:
                        sentiment_dist = px.pie(
                            sentiment_df,
                            names='category',
                            title='Sentiment Distribution'
                        )
                        st.plotly_chart(sentiment_dist)
                        
                        # Display recent news articles
                        st.subheader("Recent News Articles")
                        for article in sentiment_data['articles']:
                            with st.expander(article['title']):
                                st.write(f"Sentiment: {article['sentiment']:.2f}")
                                st.write(f"Category: {article['category']}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    predictor = StockSentimentPredictor()
    predictor.create_dashboard()
