from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from engine import analyze_product_data
import io

# Initialize FastAPI with metadata for the automatic documentation
app = FastAPI(
    title="PriceOptima AI API",
    description="Enterprise Retail Intelligence Engine for Sentiment-Driven Price Optimization. Upload a CSV of product data and reviews to receive automated pricing recommendations.",
    version="1.0.0"
)

# Allow Cross-Origin Requests (CORS) so your Streamlit app can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def home():
    """Simple health check endpoint to verify the API is running."""
    return {
        "status": "online",
        "message": "PriceOptima API is running! Visit /docs for the interactive API documentation."
    }

@app.post("/analyze", tags=["Intelligence Engine"])
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    **Upload a CSV file** containing product data.
    
    Required CSV Columns:
    - `price`: Current price of the item
    - `average_rating`: Star rating (1 to 5)
    - `rating_number`: Total number of reviews
    - `comments`: The actual text of the customer reviews
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
        
    try:
        # Read the uploaded file into memory
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Run through the PriceOptima diagnostic engine
        analysis_results = analyze_product_data(df)
        
        return {
            "status": "success",
            "data": analysis_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))