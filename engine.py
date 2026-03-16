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
    raise FileNotFoundError(f"Model {MODEL_PATH} not found.")

analyzer = SentimentIntensityAnalyzer()

def analyze_product_data(df):
    # 1. Feature Engineering
    df['average_rating'] = df.get('average_rating', 4.0)
    df['rating_number'] = df.get('rating_number', 100)
    df['price'] = df.get('price', 50.0)
    df['title'] = df.get('product_name', 'Unknown Product')
    df['comments'] = df.get('comments', '').astype(str)
    
    df['net_sentiment'] = [analyzer.polarity_scores(text)['compound'] for text in df['comments']]
    
    df['review_len'] = df['comments'].str.len().fillna(0)
    df['price_log1p'] = np.log1p(np.where(df['price'] > 0, df['price'], 0))
    df['price_x_avg_rating'] = df['price'] * df['average_rating']
    
    # 2. Inference
    feature_cols = ['price', 'average_rating', 'rating_number', 'net_sentiment', 'review_len', 'price_log1p', 'price_x_avg_rating']
    X_predict = df[feature_cols].fillna(0)
    df['predicted_demand'] = rf_model.predict(X_predict)
    median_demand = df['predicted_demand'].median()
    
    # 3. EXACT Managerial Logic & A/B Testing Math
    conditions = [
        (df['predicted_demand'] > median_demand) & (df['net_sentiment'] > 0), # Star
        (df['predicted_demand'] <= median_demand) & (df['net_sentiment'] > 0), # Hidden Gem
        (df['predicted_demand'] > median_demand) & (df['net_sentiment'] <= 0), # Time Bomb
        (df['predicted_demand'] <= median_demand) & (df['net_sentiment'] <= 0)  # Dog
    ]
    
    diagnoses = ["Star", "Hidden Gem", "Ticking Time Bomb", "Dog"]
    recommendations = ["INCREASE PRICE", "DECREASE PRICE", "RETAIN & IMPROVE", "DECREASE & PHASE OUT"]
    
    # Generate A/B Testing target prices (Star: +5%, Gem: -10%, Bomb: No Change, Dog: -50%)
    ab_test_multipliers = [1.05, 0.90, 1.00, 0.50]
    
    reasonings = [
        "Positive sentiment buffer allows margin capture. Initiate a 5% A/B price test.",
        "Excellent product but low visibility. Decrease price by 10% to stimulate trial.",
        "High volume but fragile sentiment. Do not adjust price; route to QA immediately.",
        "Low demand and poor perception. Slash price by 50% for liquidation."
    ]
    
    df['diagnosis'] = np.select(conditions, diagnoses, default="Unknown")
    df['recommendation'] = np.select(conditions, recommendations, default="REVIEW MANUALLY")
    df['reasoning'] = np.select(conditions, reasonings, default="Unclassifiable.")
    df['ab_test_price'] = (df['price'] * np.select(conditions, ab_test_multipliers, default=1.0)).round(2)
    
    # 4. Clean and Format
    df['product_name'] = df['title']
    df['current_price'] = df['price'].round(2)
    df['net_sentiment'] = np.round(df['net_sentiment'], 3)
    df['monthly_sales'] = np.round(df['predicted_demand'], 2)
    
    return df[['product_name', 'current_price', 'ab_test_price', 'net_sentiment', 'monthly_sales', 'diagnosis', 'recommendation', 'reasoning']].to_dict(orient='records')