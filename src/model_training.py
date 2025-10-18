import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def train_isolation_forest(X_train, y_train):
    """
    Train Isolation Forest for anomaly detection (unsupervised)
    """
    print("\n" + "="*60)
    print("🌲 TRAINING ISOLATION FOREST (Anomaly Detection)")
    print("="*60)
    
    # Isolation Forest detects anomalies without labels
    iso_forest = IsolationForest(
        contamination=0.002,  # Expected fraud rate
        random_state=42,
        n_estimators=100,
        max_samples=256,
        n_jobs=-1
    )
    
    print("   Training on normal transactions only...")
    # Train only on normal transactions (unsupervised learning)
    X_train_normal = X_train[y_train == 0]
    iso_forest.fit(X_train_normal)
    
    print(f"   ✅ Model trained on {len(X_train_normal):,} normal transactions")
    
    return iso_forest

def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier (supervised)
    """
    print("\n" + "="*60)
    print("🚀 TRAINING XGBOOST (Supervised Learning)")
    print("="*60)
    
    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Class imbalance ratio: {scale_pos_weight:.0f}:1")
    print(f"   Applying scale_pos_weight: {scale_pos_weight:.2f}")
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='aucpr',  # Focus on precision-recall
        use_label_encoder=False
    )
    
    print("   Training XGBoost model...")
    xgb_model.fit(X_train, y_train)
    
    print(f"   ✅ Model trained on {len(X_train):,} transactions")
    
    return xgb_model

def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*60)
    print(f"📊 EVALUATING {model_name}")
    print("="*60)
    
    # Get predictions
    if model_name == "Isolation Forest":
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = model.predict(X_test)
        y_pred = (predictions == -1).astype(int)
        # Get anomaly scores (more negative = more anomalous)
        y_scores = -model.score_samples(X_test)
    else:
        # XGBoost probability predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        y_scores = y_proba
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    
    # Precision-Recall AUC (better for imbalanced data)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Precision: {precision:.4f} (of flagged transactions, how many are actual fraud)")
    print(f"   Recall: {recall:.4f} (of all frauds, how many did we catch)")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC: {pr_auc:.4f} (better metric for imbalanced data)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"   True Negatives (Correct normal):  {tn:,}")
    print(f"   False Positives (False alarms):   {fp:,}")
    print(f"   False Negatives (Missed frauds):  {fn:,}")
    print(f"   True Positives (Caught frauds):   {tp:,}")
    
    # Business metrics
    false_positive_rate = fp / (fp + tn)
    print(f"\n💼 BUSINESS METRICS:")
    print(f"   False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    print(f"   → {fp:,} innocent customers flagged out of {fp+tn:,}")
    print(f"   Fraud Detection Rate: {recall:.4f} ({recall*100:.2f}%)")
    print(f"   → Caught {tp} out of {tp+fn} frauds")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cm': cm,
        'y_pred': y_pred,
        'y_scores': y_scores
    }

def create_ensemble_predictions(iso_results, xgb_results, weights=(0.3, 0.7)):
    """
    Combine predictions from both models
    """
    print("\n" + "="*60)
    print("🎭 CREATING ENSEMBLE PREDICTIONS")
    print("="*60)
    
    # Normalize scores to 0-1 range
    iso_scores_norm = (iso_results['y_scores'] - iso_results['y_scores'].min()) / \
                      (iso_results['y_scores'].max() - iso_results['y_scores'].min())
    
    # Weighted average
    ensemble_scores = weights[0] * iso_scores_norm + weights[1] * xgb_results['y_scores']
    ensemble_pred = (ensemble_scores >= 0.5).astype(int)
    
    print(f"   Isolation Forest weight: {weights[0]}")
    print(f"   XGBoost weight: {weights[1]}")
    
    return ensemble_scores, ensemble_pred

if __name__ == "__main__":
    print("="*60)
    print("🚀 FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load processed data
    print("\n📂 Loading processed data...")
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    print(f"   ✅ Training set: {len(X_train):,} samples")
    print(f"   ✅ Test set: {len(X_test):,} samples")
    
    # Train Isolation Forest
    iso_model = train_isolation_forest(X_train, y_train)
    
    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    
    # Evaluate Isolation Forest
    iso_results = evaluate_model(iso_model, X_test, y_test, "Isolation Forest")
    
    # Evaluate XGBoost
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # Create ensemble
    ensemble_scores, ensemble_pred = create_ensemble_predictions(iso_results, xgb_results)
    
    # Evaluate ensemble
    ensemble_precision = precision_score(y_test, ensemble_pred)
    ensemble_recall = recall_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred)
    
    print(f"\n📈 ENSEMBLE PERFORMANCE:")
    print(f"   Precision: {ensemble_precision:.4f}")
    print(f"   Recall: {ensemble_recall:.4f}")
    print(f"   F1-Score: {ensemble_f1:.4f}")
    
    # Save models
    print("\n💾 Saving models...")
    joblib.dump(iso_model, 'models/isolation_forest.pkl')
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    
    # Save metadata
    model_metadata = {
        'iso_metrics': {
            'precision': iso_results['precision'],
            'recall': iso_results['recall'],
            'f1': iso_results['f1'],
            'pr_auc': iso_results['pr_auc']
        },
        'xgb_metrics': {
            'precision': xgb_results['precision'],
            'recall': xgb_results['recall'],
            'f1': xgb_results['f1'],
            'pr_auc': xgb_results['pr_auc']
        },
        'ensemble_metrics': {
            'precision': float(ensemble_precision),
            'recall': float(ensemble_recall),
            'f1': float(ensemble_f1)
        },
        'feature_count': X_train.shape[1],
        'training_samples': len(X_train)
    }
    
    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"   ✅ Isolation Forest saved: models/isolation_forest.pkl")
    print(f"   ✅ XGBoost saved: models/xgboost_model.pkl")
    print(f"   ✅ Metadata saved: models/model_metadata.json")
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\n🎯 RECOMMENDED FOR RESUME:")
    print(f"   'Built ensemble fraud detection with {ensemble_precision:.1%} precision,")
    print(f"    {ensemble_recall:.1%} recall, detecting fraud with <0.X% false positive rate'")