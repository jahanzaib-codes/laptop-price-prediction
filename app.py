"""
Laptop Price Prediction Web Application
Created by: Jahanzaib
A Flask-based web application for predicting laptop prices using ML and Gemini AI

Features:
- Manual specification-based price prediction
- AI-powered image-based prediction using Google Gemini
- RESTful API endpoints
- Modern animated web interface
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_API_KEY_HERE':
    genai.configure(api_key=GEMINI_API_KEY)
    # Using gemini-2.5-flash - latest model
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini AI configured with model: gemini-2.5-flash")
else:
    gemini_model = None
    print("⚠️ Warning: GEMINI_API_KEY not found or invalid. AI features will not work.")
    print("   Get your API key from: https://makersuite.google.com/app/apikey")

# Model paths
MODEL_PATH = 'models/laptop_price_model.joblib'
ENCODERS_PATH = 'models/label_encoders.joblib'
SCALER_PATH = 'models/scaler.joblib'
FEATURE_COLUMNS_PATH = 'models/feature_columns.joblib'

# Load ML model and preprocessors
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURE_COLUMNS_PATH)
        print("✅ Model and preprocessors loaded successfully!")
        return model, encoders, scaler, feature_cols
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run train_model.py first to train the model.")
        return None, None, None, None

# Initialize model
ml_model, encoders, scaler, feature_cols = load_model()

# Valid options for dropdowns
VALID_OPTIONS = {
    'Company': ['Dell', 'HP', 'Lenovo', 'Apple', 'Asus', 'Acer', 'MSI'],
    'TypeName': ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation'],
    'Cpu_brand': ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9', 
                  'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9',
                  'Apple M1', 'Apple M1 Pro', 'Apple M1 Max', 'Apple M2', 'Apple M2 Pro'],
    'Gpu_brand': ['Intel', 'Nvidia', 'AMD', 'Apple'],
    'os': ['Windows', 'Mac', 'Linux'],
    'Ram': [4, 8, 16, 32, 64],
    'SSD': [0, 128, 256, 512, 1024, 2048],
    'HDD': [0, 256, 500, 1000, 2000]
}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/options', methods=['GET'])
def get_options():
    """Return valid options for form fields"""
    return jsonify(VALID_OPTIONS)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict laptop price based on specifications
    
    Request body:
    {
        "Company": "Dell",
        "TypeName": "Ultrabook",
        "Ram": 8,
        "Weight": 1.5,
        "Touchscreen": 0,
        "Ips": 1,
        "ppi": 226.98,
        "Cpu_brand": "Intel Core i5",
        "HDD": 0,
        "SSD": 256,
        "Gpu_brand": "Intel",
        "os": "Windows"
    }
    """
    if ml_model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.json
        
        # Create DataFrame with input data
        input_df = pd.DataFrame([data])
        
        # Define categorical and numerical columns
        categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'os']
        numerical_cols = ['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in input_df.columns and col in encoders:
                # Handle unknown categories
                try:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except ValueError:
                    # If unknown category, use the first known category
                    input_df[col] = 0
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select and order columns
        input_df = input_df[feature_cols]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = ml_model.predict(input_scaled)[0]
        
        # Round to nearest thousand
        prediction = round(prediction / 1000) * 1000
        
        # Generate insights
        insights = generate_insights(data, prediction)
        
        return jsonify({
            'success': True,
            'predicted_price': int(prediction),
            'price_formatted': f"Rs. {int(prediction):,}",
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def generate_insights(specs, price):
    """Generate insights about the prediction"""
    insights = []
    
    # RAM insights
    ram = int(specs.get('Ram', 8))
    if ram >= 32:
        insights.append("🚀 High RAM (32GB+) is excellent for multitasking, video editing, and heavy workloads.")
    elif ram >= 16:
        insights.append("💪 16GB RAM is perfect for gaming, development, and professional work.")
    elif ram >= 8:
        insights.append("👍 8GB RAM is good for everyday tasks, light gaming, and office work.")
    else:
        insights.append("⚠️ Consider upgrading RAM for better performance in modern applications.")
    
    # SSD insights
    ssd = int(specs.get('SSD', 0))
    if ssd >= 512:
        insights.append("⚡ 512GB+ SSD provides fast boot times and ample storage.")
    elif ssd >= 256:
        insights.append("✓ 256GB SSD is good for OS and essential apps.")
    elif ssd == 0:
        insights.append("💡 Consider an SSD for significantly faster performance.")
    
    # GPU insights
    gpu = specs.get('Gpu_brand', 'Intel')
    if gpu == 'Nvidia':
        insights.append("🎮 NVIDIA GPU is great for gaming, AI/ML, and creative work.")
    elif gpu == 'AMD':
        insights.append("🖥️ AMD GPU offers good value for gaming and productivity.")
    elif gpu == 'Apple':
        insights.append("🍎 Apple Silicon provides excellent efficiency and performance.")
    
    # Price category
    if price < 40000:
        insights.append("💰 Budget-friendly laptop suitable for basic tasks.")
    elif price < 80000:
        insights.append("📊 Mid-range laptop with good balance of features and price.")
    elif price < 150000:
        insights.append("🏆 Premium laptop with high-end specifications.")
    else:
        insights.append("👑 Ultra-premium laptop with top-of-the-line specifications.")
    
    return insights

@app.route('/api/predict-image', methods=['POST'])
def predict_from_image():
    """
    Predict laptop specs and price from an image using Gemini AI
    
    Request body:
    {
        "image": "base64_encoded_image_string"
    }
    """
    if gemini_model is None:
        return jsonify({
            'success': False,
            'error': 'Gemini AI not configured. Please check your API key.'
        }), 500
    
    try:
        data = request.json
        image_data = data.get('image', '')
        
        # Handle data URL format
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Prepare prompt for Gemini
        prompt = """Analyze this laptop image and extract the following specifications. 
        If you cannot determine a specification from the image, make an educated guess based on the laptop's appearance, brand, and typical configurations.
        
        Please respond ONLY with a valid JSON object in this exact format (no markdown, no code blocks, just raw JSON):
        {
            "brand": "detected or estimated brand name",
            "model": "detected or estimated model name",
            "estimated_specs": {
                "Company": "brand name (Dell/HP/Lenovo/Apple/Asus/Acer/MSI)",
                "TypeName": "type (Ultrabook/Notebook/Gaming/2 in 1 Convertible)",
                "Ram": estimated RAM in GB (number only: 4/8/16/32/64),
                "Weight": estimated weight in kg (number like 1.5),
                "Touchscreen": 0 or 1,
                "Ips": 0 or 1,
                "ppi": estimated pixels per inch (number like 141.21),
                "Cpu_brand": "processor type (Intel Core i3/i5/i7/i9 or AMD Ryzen 3/5/7/9 or Apple M1/M2)",
                "HDD": HDD size in GB (0/500/1000/2000),
                "SSD": SSD size in GB (0/128/256/512/1024),
                "Gpu_brand": "GPU brand (Intel/Nvidia/AMD/Apple)",
                "os": "operating system (Windows/Mac/Linux)"
            },
            "confidence": "low/medium/high",
            "description": "brief description of what you observed in the image"
        }"""
        
        # Call Gemini API
        response = gemini_model.generate_content([prompt, image])
        response_text = response.text.strip()
        
        # Clean the response - remove markdown code blocks if present
        if response_text.startswith('```'):
            # Remove ```json or ``` from start and ``` from end
            lines = response_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1].strip() == '```':
                lines = lines[:-1]
            response_text = '\n'.join(lines)
        
        # Parse Gemini response
        ai_analysis = json.loads(response_text)
        
        # Use extracted specs for prediction
        specs = ai_analysis.get('estimated_specs', {})
        
        if ml_model is not None and specs:
            # Create prediction using ML model
            input_df = pd.DataFrame([specs])
            
            categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'os']
            
            for col in categorical_cols:
                if col in input_df.columns and col in encoders:
                    try:
                        input_df[col] = encoders[col].transform(input_df[col].astype(str))
                    except ValueError:
                        input_df[col] = 0
            
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[feature_cols]
            input_scaled = scaler.transform(input_df)
            prediction = ml_model.predict(input_scaled)[0]
            prediction = round(prediction / 1000) * 1000
            
            ai_analysis['predicted_price'] = int(prediction)
            ai_analysis['price_formatted'] = f"Rs. {int(prediction):,}"
        
        return jsonify({
            'success': True,
            'analysis': ai_analysis
        })
        
    except json.JSONDecodeError as e:
        return jsonify({
            'success': False,
            'error': f'Failed to parse AI response: {str(e)}',
            'raw_response': response_text if 'response_text' in locals() else 'No response'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/gemini-info', methods=['POST'])
def get_laptop_info():
    """
    Get detailed information about a laptop using Gemini AI
    
    Request body:
    {
        "query": "Tell me about Dell XPS 15"
    }
    """
    if gemini_model is None:
        return jsonify({
            'success': False,
            'error': 'Gemini AI not configured. Please check your API key.'
        }), 500
    
    try:
        data = request.json
        query = data.get('query', '')
        
        prompt = f"""You are a helpful laptop buying assistant. Answer the following question about laptops in a detailed but easy to understand way.
        
        Question: {query}
        
        Please provide:
        1. Direct answer to the question
        2. Key specifications to consider
        3. Price range estimate (in Pakistani Rupees)
        4. Pros and cons if applicable
        5. Buying recommendations
        
        Format your response in a clear, organized manner with emojis for better readability."""
        
        response = gemini_model.generate_content(prompt)
        
        return jsonify({
            'success': True,
            'response': response.text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/help', methods=['GET'])
def get_help():
    """Return help information for finding laptop specifications"""
    help_info = {
        'title': '🔍 How to Find Laptop Specifications for Prediction',
        'methods': [
            {
                'title': '1. Check Laptop Label/Sticker',
                'description': 'Look at the bottom of your laptop or around the keyboard area for a label showing model number and basic specs.',
                'icon': '🏷️'
            },
            {
                'title': '2. System Information (Windows)',
                'description': 'Press Win + R, type "msinfo32" and press Enter. This shows detailed system information including processor, RAM, and more.',
                'icon': '💻'
            },
            {
                'title': '3. About This Mac (Apple)',
                'description': 'Click the Apple menu → About This Mac. This shows your Mac model, processor, memory, and storage.',
                'icon': '🍎'
            },
            {
                'title': '4. Device Manager',
                'description': 'Right-click Start → Device Manager. Expand categories to see detailed hardware information.',
                'icon': '⚙️'
            },
            {
                'title': '5. Task Manager (Windows)',
                'description': 'Press Ctrl + Shift + Esc → Performance tab. Shows CPU, RAM, GPU, and storage details.',
                'icon': '📊'
            },
            {
                'title': '6. Third-Party Tools',
                'description': 'Use CPU-Z, HWiNFO, or Speccy for detailed hardware specifications.',
                'icon': '🛠️'
            },
            {
                'title': '7. Manufacturer Website',
                'description': 'Search your laptop model number on the manufacturer\'s website for official specifications.',
                'icon': '🌐'
            },
            {
                'title': '8. Take a Photo!',
                'description': 'Use our AI Image Prediction feature! Just upload a photo of the laptop and our Gemini AI will analyze it.',
                'icon': '📸'
            }
        ],
        'specs_explanation': {
            'RAM': 'Random Access Memory - More RAM means better multitasking. 8GB minimum, 16GB recommended for 2024.',
            'SSD': 'Solid State Drive - Faster than HDD. 256GB minimum, 512GB recommended.',
            'HDD': 'Hard Disk Drive - Slower but cheaper storage. Good for data backup.',
            'CPU': 'Processor brand and generation. Intel Core i5/i7 or AMD Ryzen 5/7 are popular choices.',
            'GPU': 'Graphics card. Nvidia for gaming/creative work, Intel/AMD integrated for basic use.',
            'Touchscreen': 'Touch-enabled display. Useful for 2-in-1 laptops.',
            'IPS': 'In-Plane Switching display. Better colors and viewing angles.',
            'PPI': 'Pixels Per Inch. Higher means sharper display. 150+ is good.',
            'Weight': 'Laptop weight in kilograms. Under 1.5kg is ultraportable.'
        }
    }
    return jsonify(help_info)

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎓 LAPTOP PRICE PREDICTION WEB APPLICATION")
    print("   Created by: Jahanzaib")
    print("=" * 60)
    print("\n🚀 Starting server at http://localhost:5000")
    print("📝 API Endpoints:")
    print("   GET  /              - Main web interface")
    print("   GET  /api/options   - Get dropdown options")
    print("   POST /api/predict   - Predict price from specs")
    print("   POST /api/predict-image - Predict from image (AI)")
    print("   POST /api/gemini-info   - Get laptop info from AI")
    print("   GET  /api/help      - Get help information")
    print("\n")
    
    app.run(debug=True, port=5000)
