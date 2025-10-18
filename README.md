# 🚨 Real-Time Fraud Detection System

[![Azure](https://img.shields.io/badge/Deployed%20on-Azure-0078D4?logo=microsoft-azure)](https://fraud-detection-api.delightfulpebble-be710419.westus2.azurecontainerapps.io/docs)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)](https://www.docker.com/)

**Production-grade fraud detection system preventing $1.65M+ in annual fraud losses with 87% precision and 78% recall.**

🔗 **Live API:** [https://fraud-detection-api.delightfulpebble-be710419.westus2.azurecontainerapps.io/docs](https://fraud-detection-api.delightfulpebble-be710419.westus2.azurecontainerapps.io/docs)

---

## 📊 Project Highlights

- **87.1% Precision** - 87 out of 100 fraud alerts are genuine
- **77.9% Recall** - Catches 78% of all fraud cases
- **$1.65M Annual Impact** - Projected fraud prevention value
- **1,590% ROI** - Return on investment from fraud prevention
- **<100ms Latency** - Real-time transaction processing
- **Explainable AI** - SHAP values for regulatory compliance

---

## 🏗️ Architecture
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Client    │─────▶│  FastAPI     │─────▶│  Ensemble       │
│ Application │      │  REST API    │      │  ML Models      │
└─────────────┘      └──────────────┘      └─────────────────┘
│                       │
│                       ├─ Isolation Forest
│                       └─ XGBoost
▼
┌──────────────┐
│   Azure      │
│ Container    │
│   Apps       │
└──────────────┘

**Technology Stack:**
- **ML Framework:** scikit-learn, XGBoost, SHAP
- **API:** FastAPI with async support
- **Deployment:** Docker + Azure Container Apps
- **Monitoring:** Azure Log Analytics
- **Data Processing:** Pandas, NumPy

---

## 🎯 Business Impact

### Financial Metrics
- **Annual Fraud Prevented:** $1,650,446
- **ROI:** 1,590%
- **Detection Rate:** 77.9%
- **False Positive Rate:** 0.02%

### Model Performance
| Metric | Value |
|--------|-------|
| Precision | 87.1% |
| Recall | 77.9% |
| F1-Score | 0.822 |
| PR-AUC | 0.856 |

---

## 🔬 Technical Approach

### 1. Feature Engineering
Created 18+ advanced features from raw transaction data:

- **Time-based:** Cyclical hour encoding, day patterns
- **Amount-based:** Log transformation, percentile ranking, decimal patterns
- **Statistical:** Aggregations from PCA components (V1-V28)
- **Interactions:** Cross-feature relationships

**Key Insight:** Our engineered features (`v_std`, `v4_amount_interaction`) ranked in top 10 most important!

### 2. Ensemble Model
Combined two complementary approaches:

**Isolation Forest (30% weight)**
- Unsupervised anomaly detection
- Identifies unusual transaction patterns
- No fraud labels required

**XGBoost (70% weight)**
- Supervised classification
- Handles extreme class imbalance (577:1)
- Uses `scale_pos_weight` for optimization

### 3. Explainability (Regulatory Compliance)
- **SHAP values** for every prediction
- **Feature importance** rankings
- **Waterfall plots** showing decision logic
- Can explain why any transaction was flagged

---

## 🚀 Quick Start

### Test the Live API
```bash
# Health check
curl https://fraud-detection-api.delightfulpebble-be710419.westus2.azurecontainerapps.io/health

# Interactive docs
open https://fraud-detection-api.delightfulpebble-be710419.westus2.azurecontainerapps.io/docs
Run Locally
bash# Clone repository
git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API server
python src/api.py
Visit: http://localhost:8000/docs
Docker
bash# Build image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 fraud-detection-api

📁 Project Structure
fraud-detection-system/
├── src/
│   ├── api.py                      # FastAPI production service
│   ├── feature_engineering.py      # Feature creation pipeline
│   ├── model_training.py           # Ensemble model training
│   ├── model_explainability.py     # SHAP analysis & ROI calc
│   └── monitoring.py               # Performance dashboard
├── models/
│   ├── xgboost_model.pkl          # Trained XGBoost classifier
│   ├── isolation_forest.pkl       # Trained Isolation Forest
│   ├── scaler.pkl                 # Feature scaler
│   ├── shap_explainer.pkl         # SHAP explainer
│   ├── business_impact.json       # ROI metrics
│   └── monitoring_dashboard.png   # Performance viz
├── data/
│   └── creditcard.csv             # Kaggle credit card dataset
├── Dockerfile                      # Container configuration
├── requirements.txt                # Python dependencies
└── README.md

📊 Key Features
Real-Time Prediction API
pythonPOST /predict
{
  "Time": 12345.0,
  "V1": -1.359807,
  "Amount": 149.62,
  ...
}

Response:
{
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "risk_level": "LOW",
  "processing_time_ms": 45.2
}
