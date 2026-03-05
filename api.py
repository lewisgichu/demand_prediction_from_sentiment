from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from engine import analyze_product_data

app = Flask(__name__)
CORS(app) # Allows cross-origin requests

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "PriceOptima API is running! Send a POST request with a CSV file to the /analyze endpoint."
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Run through our diagnostic engine
        analysis_results = analyze_product_data(df)
        
        return jsonify({
            "status": "success",
            "data": analysis_results
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask API on port 5000
    app.run(debug=True, port=5000)