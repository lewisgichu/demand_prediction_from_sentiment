import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import time

print("Initializing PriceOptima Data Science Pipeline...")
start_time = time.time()

# 1. Load or Generate Data
# In production, you would load the Amazon JSONL files here. 
# For local operationalization, we generate a synthetic dataset matching your EDA distributions.
np.random.seed(42)
n_samples = 5000

data = {
    'price': np.random.uniform(5.0, 350.0, n_samples),
    'average_rating': np.random.uniform(1.0, 5.0, n_samples),
    'rating_number': np.random.randint(1, 5000, n_samples),
    'title': ["Smartphone case " * np.random.randint(1, 5) for _ in range(n_samples)],
    'text': ["Great product, really love the quality but price is a bit high." for _ in range(n_samples)]
}
df = pd.DataFrame(data)

# 2. Feature Engineering (Replicating notebook logic)
print("Running Feature Engineering & NLP Sentiment Extraction...")
analyzer = SentimentIntensityAnalyzer()

def safe_sentiment(text):
    try:
        return analyzer.polarity_scores(str(text))['compound']
    except:
        return 0.0

# Extract VADER Sentiment
df['net_sentiment'] = df['text'].apply(safe_sentiment)

# Advanced features from your notebook
df['title_len'] = df['title'].apply(lambda x: len(str(x)))
df['price_log1p'] = df['price'].apply(lambda x: math.log1p(x) if x > 0 else 0)
df['price_x_avg_rating'] = df['price'] * df['average_rating']
df['rating_number_log1p'] = df['rating_number'].apply(lambda x: math.log1p(x))

# Define Target Variable (Demand Proxy as per your notebook)
# Simulated relationship where lower price, higher rating, and positive sentiment drive demand
df['demand_proxy'] = (df['rating_number_log1p'] * df['average_rating']) + (df['net_sentiment'] * 2) - df['price_log1p']

# Select final features
feature_cols = ['price', 'average_rating', 'rating_number', 'net_sentiment', 
                'title_len', 'price_log1p', 'price_x_avg_rating']

X = df[feature_cols]
y = df['demand_proxy']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training (Random Forest)
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. Model Evaluation (Aligning with Chapter 5 of your Proposal)
preds = rf_model.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("\n" + "="*40)
print("MODEL EVALUATION METRICS")
print("="*40)
print(f"R-Squared (R2): {r2:.4f}")
print(f"RMSE:           {rmse:.4f}")
print(f"MAE:            {mae:.4f}")
print("="*40)

# 6. Save the Model Artifact
model_path = 'rf_model_v1.joblib'
joblib.dump(rf_model, model_path)
print(f"\nModel successfully saved to {model_path} in {time.time() - start_time:.2f} seconds.")