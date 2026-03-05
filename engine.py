import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_product_data(df):
    results = []
    
    # Calculate median demand for the "SubCategory" threshold mentioned in your paper
    median_demand = df['monthly_sales'].median() if 'monthly_sales' in df.columns else 100
    
    for index, row in df.iterrows():
        # 1. Analyze Sentiment using VADER
        comments = str(row.get('comments', ''))
        sentiment_scores = analyzer.polarity_scores(comments)
        net_sentiment = sentiment_scores['compound'] # Ranges from -1.0 to 1.0
        
        # 2. Determine Demand
        sales = float(row.get('monthly_sales', 0))
        demand_status = "High Demand" if sales > median_demand else "Low Demand"
        sentiment_status = "Positive Sentiment" if net_sentiment > 0 else "Negative Sentiment"
        
        # 3. Apply the Diagnostic Logic (From Chapter 3.8.1 of your PDF)
        if demand_status == "High Demand" and sentiment_status == "Positive Sentiment":
            diagnosis = "Star"
            recommendation = "INCREASE"
            reasoning = "Customers love it and are buying it. Increase price to capture consumer surplus."
        elif demand_status == "Low Demand" and sentiment_status == "Positive Sentiment":
            diagnosis = "Hidden Gem"
            recommendation = "DECREASE"
            reasoning = "Few customers find it, but they love it. Temporary price drop acts as a marketing lever."
        elif demand_status == "High Demand" and sentiment_status == "Negative Sentiment":
            diagnosis = "Ticking Time Bomb"
            recommendation = "RETAIN & IMPROVE"
            reasoning = "People buy it but dislike it. Retain price while improving quality to prevent demand collapse."
        else:
            diagnosis = "Dog"
            recommendation = "DECREASE & PHASE OUT"
            reasoning = "Few customers buy it, and reviews are bad. Decrease price temporarily, then remove from inventory."
            
        results.append({
            "product_name": row.get('product_name', f'Product {index}'),
            "current_price": row.get('price', 0),
            "net_sentiment": round(net_sentiment, 2),
            "monthly_sales": sales,
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "reasoning": reasoning
        })
        
    return results