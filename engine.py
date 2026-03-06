import pandas as pd
import numpy as np
import math
import joblib
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the Operationalized Model
MODEL_PATH = 'rf_model_v1.joblib'
if os.path.exists(MODEL_PATH):
    rf_model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model {MODEL_PATH} not found. Please run 'python train_model.py' first.")

analyzer = SentimentIntensityAnalyzer()

def analyze_product_data(df):
    results = []
    
    # 1. Feature Engineering (Must perfectly match train_model.py)
    df['average_rating'] = df.get('average_rating', 4.0)
    df['rating_number'] = df.get('rating_number', 100)
    df['price'] = df.get('price', 50.0)
    df['title'] = df.get('product_name', 'Unknown Product')
    
    # NLP Extraction
    df['net_sentiment'] = df['comments'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    
    # Engineered Math Features
    df['title_len'] = df['title'].apply(lambda x: len(str(x)))
    df['price_log1p'] = df['price'].apply(lambda x: math.log1p(x) if x > 0 else 0)
    df['price_x_avg_rating'] = df['price'] * df['average_rating']
    
    # 2. Inference: Predict the Demand Proxy
    feature_cols = ['price', 'average_rating', 'rating_number', 'net_sentiment', 
                    'title_len', 'price_log1p', 'price_x_avg_rating']
    
    X_predict = df[feature_cols].fillna(0)
    df['predicted_demand'] = rf_model.predict(X_predict)
    
    # Determine Portfolio Thresholds
    median_demand = df['predicted_demand'].median()
    
    # 3. Diagnostic & Strategy Logic
    for index, row in df.iterrows():
        net_sentiment = row['net_sentiment']
        demand = row['predicted_demand']
        
        demand_status = "High Demand" if demand > median_demand else "Low Demand"
        sentiment_status = "Positive Sentiment" if net_sentiment > 0 else "Negative Sentiment"
        
        if demand_status == "High Demand" and sentiment_status == "Positive Sentiment":
            diagnosis = "Star"
            recommendation = "INCREASE PRICE"
            reasoning = "High ML predicted demand coupled with positive NLP sentiment. Pricing power exists to capture consumer surplus."
        elif demand_status == "Low Demand" and sentiment_status == "Positive Sentiment":
            diagnosis = "Hidden Gem"
            recommendation = "DECREASE PRICE"
            reasoning = "Positive reviews but low predicted volume. A strategic price drop can trigger market penetration."
        elif demand_status == "High Demand" and sentiment_status == "Negative Sentiment":
            diagnosis = "Ticking Time Bomb"
            recommendation = "RETAIN & IMPROVE"
            reasoning = "High volume but deteriorating sentiment. Risk of demand collapse; halt price hikes and route to QA."
        else:
            diagnosis = "Dog"
            recommendation = "PHASE OUT"
            reasoning = "Low predicted demand and negative market feedback. Liquidate inventory and do not reorder."
            
        results.append({
            "product_name": row['title'],
            "current_price": round(row['price'], 2),
            "net_sentiment": round(net_sentiment, 3),
            "monthly_sales": round(demand, 2), # Representing the ML Demand Proxy
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "reasoning": reasoning
        })
        
    return results