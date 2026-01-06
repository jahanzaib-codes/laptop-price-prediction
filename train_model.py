"""
Laptop Price Prediction Model Training Script
Created by: Jahanzaib
Purpose: Train a machine learning model to predict laptop prices based on specifications

This script performs:
1. Data loading and preprocessing
2. Feature engineering
3. Model training using Random Forest Regressor
4. Model evaluation and saving
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Constants
DATA_PATH = 'data/laptop_data.csv'
MODEL_PATH = 'models/laptop_price_model.joblib'
ENCODERS_PATH = 'models/label_encoders.joblib'
SCALER_PATH = 'models/scaler.joblib'
FEATURE_COLUMNS_PATH = 'models/feature_columns.joblib'

def load_data():
    """Load and return the laptop dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} records with {len(df.columns)} features")
    return df

def preprocess_data(df):
    """
    Preprocess the data:
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    """
    print("\n🔧 Preprocessing data...")
    
    # Create a copy
    df_processed = df.copy()
    
    # Define categorical and numerical columns
    categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'os']
    numerical_cols = ['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']
    
    # Initialize encoders dictionary
    encoders = {}
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
            print(f"  ✓ Encoded {col}: {len(le.classes_)} unique values")
    
    # Handle missing values
    df_processed = df_processed.fillna(0)
    
    # Prepare features and target
    feature_cols = categorical_cols + numerical_cols
    feature_cols = [col for col in feature_cols if col in df_processed.columns]
    
    X = df_processed[feature_cols]
    y = df_processed['Price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print(f"✅ Preprocessing complete! Features: {len(feature_cols)}")
    
    return X_scaled, y, encoders, scaler, feature_cols

def train_model(X, y):
    """Train the Random Forest model"""
    print("\n🤖 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📈 Model Performance Metrics:")
    print(f"  Mean Absolute Error (MAE): Rs. {mae:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): Rs. {rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Feature importance
    print("\n🎯 Top 5 Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model

def save_model(model, encoders, scaler, feature_cols):
    """Save the trained model and preprocessing objects"""
    print("\n💾 Saving model and preprocessors...")
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"  ✓ Model saved to {MODEL_PATH}")
    
    # Save encoders
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"  ✓ Encoders saved to {ENCODERS_PATH}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Scaler saved to {SCALER_PATH}")
    
    # Save feature columns
    joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
    print(f"  ✓ Feature columns saved to {FEATURE_COLUMNS_PATH}")
    
    print("\n✅ All components saved successfully!")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🎓 LAPTOP PRICE PREDICTION MODEL TRAINING")
    print("   Created by: Jahanzaib")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Preprocess
    X, y, encoders, scaler, feature_cols = preprocess_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save everything
    save_model(model, encoders, scaler, feature_cols)
    
    print("\n" + "=" * 60)
    print("🎉 Training complete! Model is ready for predictions.")
    print("=" * 60)

if __name__ == "__main__":
    main()
