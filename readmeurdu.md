# 🎓 Laptop Price Prediction System

> **Yeh Project Jahanzaib Ne Banaya Hai**  
> Machine Learning + Google Gemini AI se laptop ki price predict karne wala system

---

## 📋 Fihrist (Table of Contents)

- [Project Ke Baare Mein](#-project-ke-baare-mein)
- [Features](#-features)
- [Installation (Setup Kaise Karein)](#-installation-setup-kaise-karein)
- [Istemal (Usage)](#-istemal-usage)
- [Dataset Ki Maloomat](#-dataset-ki-maloomat)
- [Data Science Ke Sawaal Jawab](#-data-science-ke-sawaal-jawab)
- [Masail Aur Hal (Troubleshooting)](#-masail-aur-hal-troubleshooting)
- [Gemini API Ka Istemal](#-gemini-api-ka-istemal)

---

## 🎯 Project Ke Baare Mein

Yeh project aik **Laptop Price Prediction System** hai jo Machine Learning se laptop ki price predict karta hai. Iske alawa Google Gemini AI bhi use hota hai jo laptop ki tasveer se specifications detect karke price batata hai.

### Project Ka Maqsad

1. **Seekhna**: Data Science aur Machine Learning ka complete process samajhna
2. **Fayda**: Laptop lene se pehle price ka andaza lagana
3. **Nayi Soch**: ML aur AI (Gemini) ko mila kar kuch naya banana

---

## ✨ Features

### 1. Manual Prediction 📊
- Laptop ki specifications khud dalo
- Turant price prediction lo
- Laptop ke baare mein insights bhi milti hain

### 2. AI Image Prediction 📸
- Laptop ki tasveer upload karo
- Gemini AI tasveer dekh kar specifications detect karega
- Automatically price predict karega

### 3. AI Chat Assistant 💬
- Laptop ke baare mein koi bhi sawaal poochho
- Buying recommendations lo
- Different laptops compare karo

### 4. Help Center ❓
- Laptop ki specifications kaise dhundein yeh seekho
- Har specification ka matlab samjho

---

## 📥 Installation (Setup Kaise Karein)

### Zaroori Cheezein

- Python 3.9 ya usse naya version
- pip (Python packages install karne ke liye)
- Internet connection (Gemini AI ke liye)

### Step by Step Guide

1. **Project Download Karo**
   ```bash
   # Agar Git hai to:
   git clone https://github.com/jahanzaib-codes/laptop-price-prediction.git
   cd laptop-price-prediction
   
   # Ya phir ZIP download karke extract karo
   ```

2. **Virtual Environment Banao (Optional but Recommended)**
   ```bash
   python -m venv venv
   
   # Windows pe activate karne ke liye:
   venv\Scripts\activate
   
   # Mac/Linux pe:
   source venv/bin/activate
   ```

3. **Required Packages Install Karo**
   ```bash
   pip install -r requirements.txt
   ```
   
   Yeh command yeh sab packages install karegi:
   - Flask (web server)
   - scikit-learn (ML model)
   - pandas (data processing)
   - google-generativeai (Gemini AI)

4. **Environment Variables Set Karo**
   
   `.env` file mein apna Gemini API key dalo:
   ```
   GEMINI_API_KEY=apni_api_key_yahan_dalo
   ```

5. **Model Train Karo**
   ```bash
   python train_model.py
   ```
   
   Yeh command model ko train karegi. Terminal pe training progress dikhai dega.

6. **Application Chalao**
   ```bash
   python app.py
   ```

7. **Browser Mein Kholo**
   
   Yeh link kholo: `http://localhost:5000`

---

## 🚀 Istemal (Usage)

### Manual Prediction Kaise Karein

1. **Manual Prediction** tab pe jao
2. Laptop ki details bharo:
   - **Brand**: Dell, HP, Lenovo, Apple, Asus, Acer, MSI
   - **Type**: Ultrabook, Notebook, Gaming, 2-in-1
   - **Processor**: Intel ya AMD ya Apple
   - **GPU**: Graphics card ka brand
   - **OS**: Windows, Mac, ya Linux
   - **RAM**: 4GB se 64GB tak
   - **SSD/HDD**: Storage ki capacity
   - **Weight**: Kitne kg hai laptop
   - **Touchscreen**: Hai ya nahi

3. **🔮 Predict Price** button dabao
4. Price aur insights dekho

### Image Se Prediction Kaise Karein

1. **AI Image Prediction** tab pe jao
2. Laptop ki tasveer upload karo (drag & drop ya click karke)
3. **🤖 Analyze with AI** button dabao
4. Gemini AI tasveer analyze karega aur:
   - Brand detect karega
   - Specifications guess karega
   - Price predict karega

### AI Se Sawaal Kaise Poochein

1. **Ask AI** tab pe jao
2. Koi bhi sawaal likho, jaise:
   - "80,000 mein konsa laptop lena chahiye?"
   - "Dell vs HP mein konsa behtar hai?"
   - "Video editing ke liye kitni RAM chahiye?"
3. AI jawab dega

---

## 📊 Dataset Ki Maloomat

### Dataset Kahan Se Milega

Kaggle se download karo:
1. [Kaggle.com](https://www.kaggle.com) pe jao
2. Search karo "Laptop Price Prediction"
3. CSV file download karo
4. `data/` folder mein save karo

### Dataset Mein Kya Cheezein Hain

| Column | Matlab | Type |
|--------|--------|------|
| Company | Laptop banane wali company | Text |
| TypeName | Laptop ka type | Text |
| Ram | RAM kitni GB hai | Number |
| Weight | Kitne kg hai | Number |
| Touchscreen | Touch screen hai ya nahi | 0 ya 1 |
| Ips | IPS display hai ya nahi | 0 ya 1 |
| ppi | Screen kitni sharp hai | Number |
| Cpu_brand | Processor konsa hai | Text |
| HDD | Hard disk kitni GB | Number |
| SSD | SSD kitni GB | Number |
| Gpu_brand | Graphics card | Text |
| os | Operating System | Text |
| Price | Laptop ki price (target) | Number |

---

## 📚 Data Science Ke Sawaal Jawab

### 1. Machine Learning Kya Hai?

Machine Learning aik tareeqa hai jismein computer data se khud seekhta hai. Jaise:
- Agar hum laptop ki specifications aur price ka data dein
- Computer pattern samajh jayega
- Naye laptop ki price predict kar sakta hai

### 2. Regression Kya Hai?

Regression aik ML technique hai jo continuous values predict karti hai (jaise price). Hum laptop price predict kar rahe hain, isliye yeh regression problem hai.

### 3. Random Forest Kya Hai?

Random Forest aik algorithm hai jo bahut saare decision trees mila kar kaam karta hai:
- Har tree alag alag features dekh kar decision leta hai
- Sab trees ka combined result zyada accurate hota hai
- Isko "ensemble learning" kehte hain

### 4. Feature Engineering Kya Hai?

Feature Engineering mein hum raw data ko better form mein badal te hain:
- Text ko numbers mein convert karna (Dell → 0, HP → 1)
- Nayi features banana (jaise screen resolution se PPI nikalna)
- Missing values handle karna

### 5. Label Encoding Kya Hai?

Label Encoding mein text categories ko numbers mein badla jata hai:
```
Dell → 0
HP → 1
Lenovo → 2
Apple → 3
```
Kyunki ML models numbers samajhte hain, text nahi.

### 6. Train-Test Split Kya Hai?

Data ko do hisson mein baat te hain:
- **Training Data (80%)**: Model seekhne ke liye
- **Test Data (20%)**: Model check karne ke liye

Agar model training data pe acha kare aur test data pe bhi, to model sahi hai.

### 7. Overfitting Kya Hai?

Jab model training data ratta maar le lekin naye data pe kharab perform kare, use overfitting kehte hain. Isko rokne ke liye:
- Cross-validation karte hain
- Model complexity kam rakhte hain
- Zyada data use karte hain

### 8. Feature Importance Kya Hai?

Feature Importance batata hai ke konsi cheez price pe sabse zyada asar karti hai. Jaise:
- RAM (bahut important)
- SSD (kaafi important)
- Brand (medium important)
- Weight (kam important)

### 9. Evaluation Metrics Kya Hain?

Model kitna acha hai yeh check karne ke liye:

- **MAE (Mean Absolute Error)**: Average galti kitni hai
  - Jaise MAE = 5000 matlab model average 5000 Rs galat predict karta hai

- **RMSE (Root Mean Squared Error)**: Badi galtiyon pe focus karta hai
  
- **R² Score**: 0 se 1 ke beech
  - R² = 0.85 matlab model 85% accurately predict karta hai

### 10. Cross-Validation Kya Hai?

Cross-validation mein data ko multiple parts mein baat ke model test karte hain. Har baar alag part test ke liye use hota hai. Isse model ki real performance pata chalti hai.

---

## 🔧 Masail Aur Hal (Troubleshooting)

### Masla 1: "Model not loaded" Error

**Wajah**: Model abhi train nahi hua

**Hal**:
```bash
python train_model.py
```

### Masla 2: "GEMINI_API_KEY not found"

**Wajah**: API key set nahi hai

**Hal**:
1. [Google AI Studio](https://makersuite.google.com/app/apikey) se API key lo
2. `.env` file mein add karo:
   ```
   GEMINI_API_KEY=tumhari_key
   ```

### Masla 3: "ModuleNotFoundError"

**Wajah**: Packages install nahi hain

**Hal**:
```bash
pip install -r requirements.txt
```

### Masla 4: Port 5000 busy hai

**Wajah**: Koi aur application port use kar rahi hai

**Hal**: `app.py` mein port badlo:
```python
app.run(debug=True, port=5001)
```

### Masla 5: Image Prediction kaam nahi kar raha

**Wajah**: Gemini API mein koi masla

**Hal**:
1. API key check karo
2. Image JPG, PNG ya WEBP mein ho
3. Internet connection check karo
4. Clear laptop image use karo

### Masla 6: Prediction accurate nahi hai

**Wajah**: Model ko zyada ya better data chahiye

**Hal**:
1. Zyada training data add karo
2. Data quality check karo
3. Different algorithm try karo

---

## 🔑 Gemini API Ka Istemal

### API Key Kaise Hasil Karein

1. [Google AI Studio](https://makersuite.google.com/app/apikey) pe jao
2. Google account se sign in karo
3. "Create API Key" pe click karo
4. Key copy karo

### API Key Lagao

`.env` file mein:
```
GEMINI_API_KEY=AIza...tumhari_key...
GEMINI_MODEL=gemini-1.5-flash
GEMINI_MAX_TOKENS=2048
GEMINI_TEMPERATURE=0.7
```

### Gemini Kya Karta Hai Is Project Mein

1. **Image Analysis**: Laptop ki tasveer se specifications nikalta hai
2. **Chat**: Laptop ke baare mein sawaalon ka jawab deta hai
3. **Recommendations**: Buying advice deta hai

### API Limits

- Free version mein limited requests hain
- Agar "rate limit exceeded" aaye to thodi der baad try karo

---

## 🏢 Generation Cores Ka Matlab

Jab model train hota hai, kuch important metrics dikhati hain:

### n_estimators (Trees)
- 100 trees matlab 100 decision trees bante hain
- Zyada trees = zyada accurate lekin slow

### max_depth (Depth)
- Har tree kitna deep ja sakta hai
- 15 depth kaafi hai laptops ke liye

### n_jobs
- -1 matlab sab CPU cores use honge
- Training fast hogi

---

## 👨‍💻 Developer

**Jahanzaib**  
Data Science Student

- GitHub: [@jahanzaib-codes](https://github.com/jahanzaib-codes)

---

## 🙏 Shukriya

- Kaggle (datasets ke liye)
- Google (Gemini AI ke liye)
- Flask community
- Scikit-learn developers

---

<div align="center">

**⭐ Agar project pasand aaya to star dena mat bhoolo! ⭐**

Pyaar se banaya ❤️ Jahanzaib ne - January 2026

</div>
