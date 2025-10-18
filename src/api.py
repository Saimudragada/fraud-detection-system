from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Dict
import time
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud detection system",
    version="1.0.0"
)

# Load models and scaler at startup
print("Loading models...")
iso_model = joblib.load('models/isolation_forest.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print("✅ Models loaded successfully!")

# Define request schema
class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed between this transaction and first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": 12345.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }

def engineer_features(transaction_dict: Dict) -> pd.DataFrame:
    """
    Apply same feature engineering as training
    """
    df = pd.DataFrame([transaction_dict])
    
    # Time-based features
    df['hour'] = (df['Time'] / 3600) % 24
    df['day'] = (df['Time'] / 86400).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Amount-based features
    df['amount_log'] = np.log1p(df['Amount'])
    
    # Simplified percentile (would use training data percentiles in production)
    df['amount_percentile'] = 2  # Placeholder
    
    df['amount_decimal'] = df['Amount'] % 1
    df['is_round_amount'] = (df['Amount'] % 1 == 0).astype(int)
    
    # Statistical features
    v_columns = [f'V{i}' for i in range(1, 29)]
    df['v_mean'] = df[v_columns].mean(axis=1)
    df['v_std'] = df[v_columns].std(axis=1)
    df['v_min'] = df[v_columns].min(axis=1)
    df['v_max'] = df[v_columns].max(axis=1)
    df['v_range'] = df['v_max'] - df['v_min']
    df['v_extreme_count'] = (np.abs(df[v_columns]) > 3).sum(axis=1)
    
    # Interaction features
    df['v1_v2_interaction'] = df['V1'] * df['V2']
    df['v4_amount_interaction'] = df['V4'] * df['amount_log']
    
    return df

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """
    Predict if a transaction is fraudulent
    
    Returns:
    - is_fraud: boolean prediction
    - fraud_probability: confidence score (0-1)
    - risk_level: LOW/MEDIUM/HIGH
    - processing_time_ms: latency
    """
    start_time = time.time()
    
    try:
        # Convert to dict and engineer features
        transaction_dict = transaction.dict()
        df_features = engineer_features(transaction_dict)
        
        # Scale features
        df_scaled = scaler.transform(df_features)
        df_scaled = pd.DataFrame(df_scaled, columns=df_features.columns)
        
        # Get predictions from both models
        # Isolation Forest
        iso_pred = iso_model.predict(df_scaled)[0]
        iso_score = -iso_model.score_samples(df_scaled)[0]
        
        # XGBoost
        xgb_proba = xgb_model.predict_proba(df_scaled)[0, 1]
        
        # Ensemble (70% XGBoost, 30% Isolation Forest weight)
        iso_score_norm = (iso_score - (-0.5)) / (0.5 - (-0.5))  # Normalize roughly
        iso_score_norm = max(0, min(1, iso_score_norm))  # Clip to 0-1
        
        ensemble_score = 0.7 * xgb_proba + 0.3 * iso_score_norm
        is_fraud = ensemble_score >= 0.5
        
        # Determine risk level
        if ensemble_score >= 0.8:
            risk_level = "HIGH"
        elif ensemble_score >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "transaction_id": f"TXN_{int(transaction.Time)}",
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(float(ensemble_score), 4),
            "risk_level": risk_level,
            "model_scores": {
                "isolation_forest": round(float(iso_score_norm), 4),
                "xgboost": round(float(xgb_proba), 4),
                "ensemble": round(float(ensemble_score), 4)
            },
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
            "amount": transaction.Amount
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_fraud_batch(transactions: list[Transaction]):
    """
    Batch prediction endpoint for multiple transactions
    """
    start_time = time.time()
    
    results = []
    for txn in transactions:
        result = await predict_fraud(txn)
        results.append(result)
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "total_transactions": len(transactions),
        "fraud_detected": sum(1 for r in results if r["is_fraud"]),
        "total_processing_time_ms": round(processing_time, 2),
        "avg_processing_time_ms": round(processing_time / len(transactions), 2),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Fraud Detection API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)