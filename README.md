# 🎓 Laptop Price Prediction System

> **A Data Science Project by Jahanzaib - January 2026**  
> Machine Learning + Google Gemini AI powered laptop price prediction system with a modern web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=for-the-badge&logo=scikit-learn)
![Gemini AI](https://img.shields.io/badge/Gemini-AI-purple?style=for-the-badge&logo=google)

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Dataset Information](#-dataset-information)
- [Model Details](#-model-details)
- [Data Science Concepts](#-data-science-concepts)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🎯 About the Project

This project is a **Laptop Price Prediction System** developed as a Data Science project. It uses **Machine Learning** to predict laptop prices based on their specifications and also integrates **Google Gemini AI** for analyzing laptop images.

### Project Purpose

1. **Educational**: Demonstrates end-to-end ML pipeline from data preprocessing to deployment
2. **Practical**: Helps users estimate laptop prices before purchasing
3. **Innovative**: Combines traditional ML with modern AI capabilities (Gemini Vision)

---

## ✨ Features

### 1. Manual Prediction 📊
- Enter laptop specifications manually
- Get instant price predictions
- Receive insights about the laptop configuration

### 2. AI Image Prediction 📸
- Upload a laptop image
- Gemini AI analyzes the laptop
- Automatically detects specifications and predicts price

### 3. AI Chat Assistant 💬
- Ask questions about laptops
- Get buying recommendations
- Compare different laptop configurations

### 4. Help Center ❓
- Learn how to find laptop specifications
- Understand what each specification means
- Get guidance for making predictions

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **Flask 3.0** | Web framework for backend |
| **Scikit-Learn** | Machine Learning model training |
| **Pandas & NumPy** | Data processing and analysis |
| **Google Gemini AI** | Image analysis and chat capabilities |
| **HTML/CSS/JS** | Frontend web interface |
| **Joblib** | Model serialization |

---

## 📥 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (optional)

### Step-by-Step Installation

1. **Clone or Download the Project**
   ```bash
   git clone https://github.com/jahanzaib-codes/laptop-price-prediction.git
   cd laptop-price-prediction
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the project root (or edit the existing one):
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-1.5-flash
   GEMINI_MAX_TOKENS=2048
   DEBUG=false
   PORT=5000
   ```

5. **Train the Machine Learning Model**
   ```bash
   python train_model.py
   ```

6. **Run the Application**
   ```bash
   python app.py
   ```

7. **Open in Browser**
   
   Visit: `http://localhost:5000`

---

## 🚀 Usage

### Using Manual Prediction

1. Go to the **Manual Prediction** tab
2. Select laptop specifications:
   - **Brand** (required): Dell, HP, Lenovo, Apple, Asus, Acer, MSI
   - **Type** (required): Ultrabook, Notebook, Gaming, 2-in-1
   - **Processor** (required): Intel Core i3/i5/i7/i9, AMD Ryzen, Apple M1/M2
   - **GPU** (required): Intel, Nvidia, AMD, Apple
   - **OS** (required): Windows, Mac, Linux
   - **RAM** (required): 4GB to 64GB
   - **SSD** (optional): 0 to 2048GB
   - **HDD** (optional): 0 to 2000GB
   - **Weight** (optional): in kilograms
   - **PPI** (optional): screen pixels per inch
   - **Touchscreen** (optional): Yes/No
   - **IPS Display** (optional): Yes/No
3. Click **🔮 Predict Price**
4. View the predicted price and insights

### Using AI Image Prediction

1. Go to the **AI Image Prediction** tab
2. Drag & drop or click to upload a laptop image
3. Click **🤖 Analyze with AI**
4. View detected specifications and predicted price

### Using AI Chat

1. Go to the **Ask AI** tab
2. Type your question about laptops
3. Get AI-powered responses with recommendations

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web interface |
| GET | `/api/options` | Get dropdown options |
| POST | `/api/predict` | Predict price from specs |
| POST | `/api/predict-image` | Predict from image (AI) |
| POST | `/api/gemini-info` | Get AI chat response |
| GET | `/api/help` | Get help information |

### Example API Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Company": "Dell",
    "TypeName": "Ultrabook",
    "Ram": 16,
    "Weight": 1.5,
    "Touchscreen": 0,
    "Ips": 1,
    "ppi": 226.98,
    "Cpu_brand": "Intel Core i7",
    "HDD": 0,
    "SSD": 512,
    "Gpu_brand": "Intel",
    "os": "Windows"
  }'
```

---

## 📊 Dataset Information

### Source
The dataset can be downloaded from Kaggle:
- [Laptop Price Prediction Dataset](https://www.kaggle.com/datasets)
- Search for "Laptop Price Prediction" on Kaggle

### Features in Dataset

| Feature | Description | Type |
|---------|-------------|------|
| Company | Laptop manufacturer | Categorical |
| TypeName | Type of laptop | Categorical |
| Ram | RAM in GB | Numerical |
| Weight | Weight in kg | Numerical |
| Touchscreen | Has touchscreen | Binary (0/1) |
| Ips | Has IPS display | Binary (0/1) |
| ppi | Screen pixels per inch | Numerical |
| Cpu_brand | Processor brand/model | Categorical |
| HDD | HDD storage in GB | Numerical |
| SSD | SSD storage in GB | Numerical |
| Gpu_brand | Graphics card brand | Categorical |
| os | Operating system | Categorical |
| Price | Laptop price (target) | Numerical |

---

## 🤖 Model Details

### Algorithm: Random Forest Regressor

**Why Random Forest?**
- Handles both numerical and categorical features well
- Robust to outliers and missing values
- Provides feature importance rankings
- Good balance of accuracy and speed

### Model Parameters

```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples in leaf
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error - Average prediction error |
| **RMSE** | Root Mean Squared Error - Penalizes large errors |
| **R² Score** | Coefficient of determination (0-1) |

---

## 📚 Data Science Concepts

### 1. What is Machine Learning?
Machine Learning is a subset of AI that enables computers to learn from data without being explicitly programmed. The model learns patterns from training data to make predictions on new data.

### 2. What is Regression?
Regression is a supervised learning technique used to predict continuous values (like prices). Our model predicts laptop prices based on their specifications.

### 3. What is Feature Engineering?
Feature engineering is the process of transforming raw data into meaningful features that improve model performance. Examples:
- Encoding categorical variables (Brand → numerical values)
- Scaling numerical features
- Creating new features (like screen PPI from resolution)

### 4. What is Label Encoding?
Label Encoding converts categorical text values into numbers. For example:
- Dell → 0
- HP → 1
- Lenovo → 2

### 5. What is Train-Test Split?
We divide data into:
- **Training Set (80%)**: Used to train the model
- **Test Set (20%)**: Used to evaluate model performance

### 6. What is Overfitting?
Overfitting occurs when a model performs well on training data but poorly on new data. We prevent this using:
- Cross-validation
- Limiting tree depth
- Using ensemble methods (Random Forest)

### 7. What is Feature Importance?
Feature importance shows which features have the most impact on predictions. Higher importance = more influence on price.

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Model not loaded" Error
**Problem**: The ML model hasn't been trained yet.

**Solution**:
```bash
python train_model.py
```

#### 2. "GEMINI_API_KEY not found" Warning
**Problem**: API key for Gemini is not configured.

**Solution**:
1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it to `.env` file:
   ```
   GEMINI_API_KEY=your_key_here
   ```

#### 3. "ModuleNotFoundError" Error
**Problem**: Required packages are not installed.

**Solution**:
```bash
pip install -r requirements.txt
```

#### 4. Port 5000 Already in Use
**Problem**: Another application is using port 5000.

**Solution**: Change the port in `app.py`:
```python
app.run(debug=True, port=5001)
```

#### 5. Image Prediction Not Working
**Problem**: Gemini API returning errors.

**Solutions**:
1. Check if API key is valid
2. Ensure image is in JPG, PNG, or WEBP format
3. Check internet connection
4. Try with a clearer laptop image

#### 6. Low Prediction Accuracy
**Problem**: Model predictions seem inaccurate.

**Solutions**:
1. Add more training data
2. Check for data quality issues
3. Consider using more features
4. Try different ML algorithms

---

## 🔮 Future Improvements

- [ ] Add more laptop brands and models
- [ ] Implement user authentication
- [ ] Add price comparison with market data
- [ ] Create mobile-responsive design
- [ ] Add multi-language support
- [ ] Implement price trend analysis
- [ ] Add recommendation system

---

## 👨‍💻 About the Developer

**Jahanzaib**  
Data Science Student

- GitHub: [@jahanzaib-codes](https://github.com/jahanzaib-codes)

---

## 📄 License

This project is created for educational purposes. Feel free to use and modify it for learning.

---

## 🙏 Acknowledgments

- Kaggle for providing datasets
- Google for Gemini AI API
- Scikit-learn community
- Flask documentation

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ by Jahanzaib

</div>
