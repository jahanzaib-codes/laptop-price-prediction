import streamlit as st
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

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Laptop Price Predictor | By Jahanzaib",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animated gradient background */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        color: #a0a0c0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Price Display */
    .price-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .price-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .price-value {
        color: white;
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 800;
    }
    
    /* Insight Cards */
    .insight-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #e0e0e0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #a0a0c0;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* Chat Message Styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    
    .bot-message {
        background: rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    
    /* Help Card */
    .help-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .help-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .help-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .help-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .help-desc {
        color: #a0a0c0;
        font-size: 0.9rem;
    }
    
    /* Spec Badge */
    .spec-badge {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .spec-name {
        color: #667eea;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .spec-desc {
        color: #a0a0c0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Banner Animation */
    .banner {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 100%;
        animation: banner-gradient 10s linear infinite;
        padding: 0.75rem;
        text-align: center;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    @keyframes banner-gradient {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }
    
    /* AI Result Card */
    .ai-result {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #a0a0c0;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
    }
    
    /* Checkbox Styling */
    .stCheckbox > label {
        color: #e0e0e0 !important;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #e0e0e0 !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(40, 167, 69, 0.2);
        border: 1px solid rgba(40, 167, 69, 0.5);
        border-radius: 10px;
    }
    
    .stError {
        background: rgba(220, 53, 69, 0.2);
        border: 1px solid rgba(220, 53, 69, 0.5);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Configure AI - Support both Streamlit Cloud secrets and local .env
# Priority: 1. Streamlit secrets (for cloud deployment)
#           2. Environment variable (for local development)
try:
    # Try Streamlit Cloud secrets first
    API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    API_KEY = None

# Fallback to environment variable
if not API_KEY:
    API_KEY = os.getenv('GEMINI_API_KEY')

if API_KEY and API_KEY != 'YOUR_API_KEY_HERE':
    genai.configure(api_key=API_KEY)
    ai_model = genai.GenerativeModel('gemini-2.5-flash')
    ai_available = True
else:
    ai_model = None
    ai_available = False

# Model paths
MODEL_PATH = 'models/laptop_price_model.joblib'
ENCODERS_PATH = 'models/label_encoders.joblib'
SCALER_PATH = 'models/scaler.joblib'
FEATURE_COLUMNS_PATH = 'models/feature_columns.joblib'

# Load ML model and preprocessors
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURE_COLUMNS_PATH)
        return model, encoders, scaler, feature_cols, True
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, False

# Initialize model
ml_model, encoders, scaler, feature_cols, model_loaded = load_model()

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

def predict_price(specs):
    """Predict laptop price based on specifications"""
    if not model_loaded:
        return None, "Model not loaded"
    
    try:
        # Create DataFrame with input data
        input_df = pd.DataFrame([specs])
        
        # Define categorical and numerical columns
        categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'os']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in input_df.columns and col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except ValueError:
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
        
        return int(prediction), None
    except Exception as e:
        return None, str(e)

def analyze_image_with_ai(image):
    """Analyze laptop image using AI"""
    if not ai_available:
        return None, "AI not configured. Please check your API key."
    
    try:
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
        
        response = ai_model.generate_content([prompt, image])
        response_text = response.text.strip()
        
        # Clean the response
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1].strip() == '```':
                lines = lines[:-1]
            response_text = '\n'.join(lines)
        
        ai_analysis = json.loads(response_text)
        return ai_analysis, None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse AI response: {str(e)}"
    except Exception as e:
        return None, str(e)

def get_ai_response(query):
    """Get AI response for laptop questions"""
    if not ai_available:
        return None, "AI not configured. Please check your API key."
    
    try:
        prompt = f"""You are a helpful laptop buying assistant. Answer the following question about laptops in a detailed but easy to understand way.
        
        Question: {query}
        
        Please provide:
        1. Direct answer to the question
        2. Key specifications to consider
        3. Price range estimate (in Pakistani Rupees)
        4. Pros and cons if applicable
        5. Buying recommendations
        
        Format your response in a clear, organized manner with emojis for better readability."""
        
        response = ai_model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

# Help information
HELP_INFO = {
    'methods': [
        {'icon': '🏷️', 'title': '1. Check Laptop Label/Sticker', 
         'description': 'Look at the bottom of your laptop or around the keyboard area for a label showing model number and basic specs.'},
        {'icon': '💻', 'title': '2. System Information (Windows)', 
         'description': 'Press Win + R, type "msinfo32" and press Enter. This shows detailed system information.'},
        {'icon': '🍎', 'title': '3. About This Mac (Apple)', 
         'description': 'Click the Apple menu → About This Mac. This shows your Mac model, processor, memory, and storage.'},
        {'icon': '⚙️', 'title': '4. Device Manager', 
         'description': 'Right-click Start → Device Manager. Expand categories to see detailed hardware information.'},
        {'icon': '📊', 'title': '5. Task Manager (Windows)', 
         'description': 'Press Ctrl + Shift + Esc → Performance tab. Shows CPU, RAM, GPU, and storage details.'},
        {'icon': '🛠️', 'title': '6. Third-Party Tools', 
         'description': 'Use CPU-Z, HWiNFO, or Speccy for detailed hardware specifications.'},
        {'icon': '🌐', 'title': '7. Manufacturer Website', 
         'description': "Search your laptop model number on the manufacturer's website for official specifications."},
        {'icon': '📸', 'title': '8. Take a Photo!', 
         'description': 'Use our AI Image Prediction feature! Just upload a photo of the laptop and our AI will analyze it.'}
    ],
    'specs': {
        'RAM': 'Random Access Memory - More RAM means better multitasking. 8GB minimum, 16GB recommended.',
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

# Main App
def main():
    # Banner
    st.markdown("""
    <div class="banner">
        ✨ This Project Was Created By Jahanzaib ✨ | 🎓 Data Science Project | 🤖 Powered by Machine Learning & AI
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">💻 Laptop Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict laptop prices accurately using Machine Learning and AI</p>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Manual Prediction", "📸 AI Image Prediction", "💬 Ask AI", "❓ Help"])
    
    # Tab 1: Manual Prediction
    with tab1:
        st.markdown("### 📊 Enter Laptop Specifications")
        st.markdown("Fill in the details below to predict the price.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            company = st.selectbox("🏢 Brand *", VALID_OPTIONS['Company'], key="company")
            gpu = st.selectbox("🎮 Graphics Card *", VALID_OPTIONS['Gpu_brand'], key="gpu")
            ssd = st.selectbox("💾 SSD (GB)", VALID_OPTIONS['SSD'], key="ssd")
        
        with col2:
            typename = st.selectbox("📱 Type *", VALID_OPTIONS['TypeName'], key="typename")
            os_type = st.selectbox("🖥️ Operating System *", VALID_OPTIONS['os'], key="os")
            hdd = st.selectbox("💿 HDD (GB)", VALID_OPTIONS['HDD'], key="hdd")
        
        with col3:
            cpu = st.selectbox("🔧 Processor *", VALID_OPTIONS['Cpu_brand'], key="cpu")
            ram = st.selectbox("🧠 RAM (GB) *", VALID_OPTIONS['Ram'], key="ram")
            weight = st.number_input("⚖️ Weight (kg)", min_value=0.5, max_value=5.0, value=1.8, step=0.1, key="weight")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            ppi = st.number_input("📺 Screen PPI", min_value=50.0, max_value=400.0, value=141.21, step=0.01, key="ppi")
        
        with col5:
            touchscreen = st.checkbox("👆 Touchscreen", key="touchscreen")
        
        with col6:
            ips = st.checkbox("✨ IPS Display", value=True, key="ips")
        
        st.markdown("")
        
        if st.button("🔮 Predict Price", key="predict_btn", use_container_width=True):
            specs = {
                'Company': company,
                'TypeName': typename,
                'Cpu_brand': cpu,
                'Gpu_brand': gpu,
                'os': os_type,
                'Ram': ram,
                'SSD': ssd,
                'HDD': hdd,
                'Weight': weight,
                'ppi': ppi,
                'Touchscreen': 1 if touchscreen else 0,
                'Ips': 1 if ips else 0
            }
            
            with st.spinner("🔄 Predicting price..."):
                price, error = predict_price(specs)
            
            if error:
                st.error(f"❌ Error: {error}")
            else:
                st.markdown(f"""
                <div class="price-display">
                    <div class="price-label">💰 Predicted Price</div>
                    <div class="price-value">Rs. {price:,}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📝 Insights")
                insights = generate_insights(specs, price)
                for insight in insights:
                    st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
    
    # Tab 2: AI Image Prediction
    with tab2:
        st.markdown("### 📸 AI Image Prediction")
        st.markdown("Upload a photo of any laptop and our AI will analyze it!")
        
        if not ai_available:
            st.warning("⚠️ AI features are not available. Please configure your API key.")
        
        uploaded_file = st.file_uploader("Upload laptop image", type=['jpg', 'jpeg', 'png', 'webp'], key="image_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Laptop Image", use_container_width=True)
            
            if st.button("🤖 Analyze with AI", key="ai_analyze_btn", use_container_width=True):
                if ai_available:
                    with st.spinner("🔄 AI is analyzing the image..."):
                        analysis, error = analyze_image_with_ai(image)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🔍 Detected Information")
                            st.markdown(f"**Brand:** {analysis.get('brand', 'Unknown')}")
                            st.markdown(f"**Model:** {analysis.get('model', 'Unknown')}")
                            st.markdown(f"**Confidence:** {analysis.get('confidence', 'Unknown')}")
                            st.markdown(f"**Description:** {analysis.get('description', 'N/A')}")
                        
                        with col2:
                            st.markdown("#### 📋 Estimated Specifications")
                            specs = analysis.get('estimated_specs', {})
                            for key, value in specs.items():
                                st.markdown(f"**{key}:** {value}")
                        
                        # Predict price with estimated specs
                        if specs:
                            price, pred_error = predict_price(specs)
                            if not pred_error and price:
                                st.markdown(f"""
                                <div class="price-display">
                                    <div class="price-label">💰 Estimated Price</div>
                                    <div class="price-value">Rs. {price:,}</div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.error("❌ AI is not configured. Please check your API key.")
    
    # Tab 3: Ask AI
    with tab3:
        st.markdown("### 💬 Ask AI")
        st.markdown("Have questions about laptops? Ask our AI assistant!")
        
        if not ai_available:
            st.warning("⚠️ AI features are not available. Please configure your API key.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": """Hello! I'm your laptop buying assistant. Ask me anything about laptops!

**Examples:**
- "Best laptop for programming under 80,000 PKR?"
- "Dell vs HP which is better?"
- "How much RAM do I need for video editing?" """}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your question here..."):
            if ai_available:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        response, error = get_ai_response(prompt)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("❌ AI is not configured. Please check your API key.")
    
    # Tab 4: Help
    with tab4:
        st.markdown("### ❓ How to Find Laptop Specifications")
        st.markdown("Learn how to find your laptop's specifications for accurate price prediction!")
        
        # Methods
        cols = st.columns(2)
        for i, method in enumerate(HELP_INFO['methods']):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="help-card">
                    <div class="help-icon">{method['icon']}</div>
                    <div class="help-title">{method['title']}</div>
                    <div class="help-desc">{method['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📚 Specification Guide")
        st.markdown("Understand what each specification means")
        
        cols = st.columns(3)
        for i, (spec, desc) in enumerate(HELP_INFO['specs'].items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="spec-badge">
                    <div class="spec-name">{spec}</div>
                    <div class="spec-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong style="background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Laptop Price Predictor</strong> | Created with ❤️ by <strong>Jahanzaib</strong></p>
        <p>Data Science Project - January 2026 | <a href="https://github.com/jahanzaib-codes" target="_blank">GitHub</a> • Powered by Machine Learning & AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
