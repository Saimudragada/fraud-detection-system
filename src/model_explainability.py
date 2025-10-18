import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from datetime import datetime

def generate_shap_explanations():
    """
    Generate SHAP explanations for model predictions
    Critical for regulatory compliance and customer disputes
    """
    print("="*60)
    print("🔍 GENERATING MODEL EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    # Load model and data
    print("\n📂 Loading models and test data...")
    xgb_model = joblib.load('models/xgboost_model.pkl')
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Sample for SHAP (using 1000 samples for speed)
    sample_size = 1000
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    print(f"   ✅ Using {sample_size} samples for SHAP analysis")
    
    # Create SHAP explainer
    print("\n🧮 Creating SHAP explainer (this takes 1-2 mins)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)
    
    print("   ✅ SHAP values computed!")
    
    # 1. GLOBAL FEATURE IMPORTANCE
    print("\n📊 Computing global feature importance...")
    
    # Get mean absolute SHAP values
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 TOP 20 MOST IMPORTANT FEATURES:")
    print(feature_importance.head(20).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    print("\n   ✅ Saved: models/feature_importance.csv")
    
    # 2. SUMMARY PLOT
    print("\n📈 Creating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values.values, X_sample, show=False, max_display=20)
    plt.title("SHAP Feature Importance - Impact on Fraud Prediction", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('models/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: models/shap_summary_plot.png")
    
    # 3. BAR PLOT
    print("\n📊 Creating feature importance bar plot...")
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("Top 20 Features by Mean |SHAP Value|", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('models/shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: models/shap_bar_plot.png")
    
    # 4. SAMPLE FRAUD EXPLANATION
    print("\n🔍 Generating explanation for a FRAUD transaction...")
    
    # Find actual fraud cases in test set
    fraud_indices = X_test[y_test == 1].index
    fraud_sample = X_test.loc[fraud_indices[0]]
    
    # Get SHAP values for this transaction
    fraud_shap = explainer(fraud_sample.to_frame().T)
    
    # Get top contributing features
    fraud_contributions = pd.DataFrame({
        'feature': X_test.columns,
        'value': fraud_sample.values,
        'shap_value': fraud_shap.values[0]
    }).sort_values('shap_value', ascending=False)
    
    print("\n🚨 FRAUD TRANSACTION EXPLANATION:")
    print(f"   Transaction Amount: ${fraud_sample['Amount']:.2f}")
    print("\n   TOP 10 FEATURES INDICATING FRAUD:")
    print(fraud_contributions.head(10).to_string(index=False))
    
    # 5. WATERFALL PLOT for fraud case
    print("\n🌊 Creating waterfall plot for fraud explanation...")
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(fraud_shap[0], max_display=15, show=False)
    plt.title("How the Model Predicted This FRAUD Transaction", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('models/fraud_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: models/fraud_waterfall_plot.png")
    
    # 6. SAMPLE NORMAL EXPLANATION
    print("\n✅ Generating explanation for a NORMAL transaction...")
    
    normal_indices = X_test[y_test == 0].index
    normal_sample = X_test.loc[normal_indices[0]]
    normal_shap = explainer(normal_sample.to_frame().T)
    
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(normal_shap[0], max_display=15, show=False)
    plt.title("How the Model Predicted This NORMAL Transaction", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('models/normal_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: models/normal_waterfall_plot.png")
    
    # Save explainer for production use
    print("\n💾 Saving SHAP explainer...")
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    print("   ✅ Saved: models/shap_explainer.pkl")
    
    print("\n" + "="*60)
    print("✅ EXPLAINABILITY ANALYSIS COMPLETE!")
    print("="*60)
    
    return feature_importance, explainer

def calculate_business_impact():
    """
    Calculate ROI and business metrics
    """
    print("\n" + "="*60)
    print("💰 CALCULATING BUSINESS IMPACT & ROI")
    print("="*60)
    
    # Load test results
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    metadata = json.load(open('models/model_metadata.json'))
    
    # Business assumptions
    AVG_FRAUD_AMOUNT = 122.21
    TOTAL_FRAUD_CASES = int(y_test.sum())
    
    # Model performance
    precision = metadata['ensemble_metrics']['precision']
    recall = metadata['ensemble_metrics']['recall']
    
    # Calculate metrics
    total_fraud_value = TOTAL_FRAUD_CASES * AVG_FRAUD_AMOUNT
    fraud_caught = int(TOTAL_FRAUD_CASES * recall)
    fraud_missed = TOTAL_FRAUD_CASES - fraud_caught
    
    value_saved = fraud_caught * AVG_FRAUD_AMOUNT
    value_lost = fraud_missed * AVG_FRAUD_AMOUNT
    
    # False positives
    total_flagged = fraud_caught / precision if precision > 0 else fraud_caught
    false_positives = int(total_flagged - fraud_caught)
    
    # Costs
    COST_PER_FALSE_POSITIVE = 15
    INVESTIGATION_COST_PER_ALERT = 5
    
    false_positive_cost = false_positives * COST_PER_FALSE_POSITIVE
    investigation_cost = fraud_caught * INVESTIGATION_COST_PER_ALERT
    
    # Net benefit
    gross_benefit = value_saved
    total_costs = false_positive_cost + investigation_cost
    net_benefit = gross_benefit - total_costs
    
    roi = (net_benefit / total_costs) * 100 if total_costs > 0 else 0
    
    print(f"\n📊 FRAUD PREVENTION METRICS:")
    print(f"   Total fraud cases: {TOTAL_FRAUD_CASES}")
    print(f"   Fraud detected: {fraud_caught} ({recall*100:.1f}%)")
    print(f"   Fraud missed: {fraud_missed}")
    
    print(f"\n💵 FINANCIAL IMPACT:")
    print(f"   Value saved: ${value_saved:,.2f}")
    print(f"   Value lost: ${value_lost:,.2f}")
    print(f"   Net benefit: ${net_benefit:,.2f}")
    print(f"   ROI: {roi:.1f}%")
    
    print(f"\n👥 CUSTOMER EXPERIENCE:")
    print(f"   False positives: {false_positives:,}")
    
    # Annual projection
    annual_multiplier = 365 / 2
    annual_fraud_prevented = value_saved * annual_multiplier
    
    print(f"\n📅 ANNUAL PROJECTION:")
    print(f"   Fraud prevented: ${annual_fraud_prevented:,.2f}")
    
    # Save
    business_metrics = {
        "fraud_detected": int(fraud_caught),
        "value_saved_usd": round(value_saved, 2),
        "roi_pct": round(roi, 1),
        "annual_fraud_prevented_usd": round(annual_fraud_prevented, 2)
    }
    
    with open('models/business_impact.json', 'w') as f:
        json.dump(business_metrics, f, indent=2)
    
    print("\n   ✅ Saved: models/business_impact.json")
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    
    return business_metrics

if __name__ == "__main__":
    feature_importance, explainer = generate_shap_explanations()
    business_metrics = calculate_business_impact()