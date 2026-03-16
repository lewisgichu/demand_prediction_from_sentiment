import pandas as pd
import numpy as np
import math
import joblib
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_PATH = 'priceoptima_xgb_tuned.joblib'
if os.path.exists(MODEL_PATH):
    rf_model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model {MODEL_PATH} not found. Please run the training script first.")

analyzer = SentimentIntensityAnalyzer()

def analyze_product_data(df):
    # 1. Feature Engineering (Vectorized)
    df['average_rating'] = df.get('average_rating', 4.0)
    df['rating_number'] = df.get('rating_number', 100)
    df['price'] = df.get('price', 50.0)
    df['title'] = df.get('product_name', 'Unknown Product')
    df['comments'] = df.get('comments', '').astype(str)
    
    # NLP Extraction (List comprehension is highly optimized in Python)
    df['net_sentiment'] = [analyzer.polarity_scores(text)['compound'] for text in df['comments']]
    
    # Engineered Math Features (Using fast Numpy arrays)
    df['review_len'] = df['comments'].str.len().fillna(0)
    df['price_log1p'] = np.log1p(np.where(df['price'] > 0, df['price'], 0))
    df['price_x_avg_rating'] = df['price'] * df['average_rating']
    
    # 2. Inference: Predict the Demand Proxy
    feature_cols = ['price', 'average_rating', 'rating_number', 'net_sentiment', 
                    'review_len', 'price_log1p', 'price_x_avg_rating']
    
    X_predict = df[feature_cols].fillna(0)
    df['predicted_demand'] = rf_model.predict(X_predict)
    
    # Determine Portfolio Thresholds
    median_demand = df['predicted_demand'].median()
    
    # 3. Diagnostic & Strategy Logic (VECTORIZED - Removes the slow 'for' loop entirely)
    conditions = [
        (df['predicted_demand'] > median_demand) & (df['net_sentiment'] > 0),
        (df['predicted_demand'] <= median_demand) & (df['net_sentiment'] > 0),
        (df['predicted_demand'] > median_demand) & (df['net_sentiment'] <= 0),
        (df['predicted_demand'] <= median_demand) & (df['net_sentiment'] <= 0)
    ]
    
    diagnoses = ["Star", "Hidden Gem", "Ticking Time Bomb", "Dog"]
    recommendations = ["INCREASE PRICE", "DECREASE PRICE", "RETAIN & IMPROVE", "PHASE OUT"]
    reasonings = [
        "High ML predicted demand coupled with positive NLP sentiment. Pricing power exists to capture consumer surplus.",
        "Positive reviews but low predicted volume. A strategic price drop can trigger market penetration.",
        "High volume but deteriorating sentiment. Risk of demand collapse; halt price hikes and route to QA.",
        "Low predicted demand and negative market feedback. Liquidate inventory and do not reorder."
    ]
    
    # Apply conditions instantly across 100,000 rows
    df['diagnosis'] = np.select(conditions, diagnoses)
    df['recommendation'] = np.select(conditions, recommendations)
    df['reasoning'] = np.select(conditions, reasonings)
    
    # 4. Clean and Format Data 
    df['product_name'] = df['title']
    df['current_price'] = df['price'].round(2)
    df['net_sentiment'] = df['net_sentiment'].round(3)
    df['monthly_sales'] = df['predicted_demand'].round(2)
    
    # Return bulk dictionary directly (Milliseconds instead of Minutes)
    return df[['product_name', 'current_price', 'net_sentiment', 'monthly_sales', 'diagnosis', 'recommendation', 'reasoning']].to_dict(orient='records')