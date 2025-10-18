import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def create_monitoring_dashboard():
    """
    Create monitoring visualizations and metrics
    """
    print("="*60)
    print("📊 CREATING MONITORING DASHBOARD")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    metadata = json.load(open('models/model_metadata.json'))
    business = json.load(open('models/business_impact.json'))
    
    # Create comprehensive dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fraud Detection System - Monitoring Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Model Performance Metrics
    ax1 = axes[0, 0]
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [
        metadata['ensemble_metrics']['precision'],
        metadata['ensemble_metrics']['recall'],
        metadata['ensemble_metrics']['f1']
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Model Performance Metrics', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Business Impact
    ax2 = axes[0, 1]
    impact_labels = ['Fraud\nPrevented', 'ROI']
    impact_values = [
        business['annual_fraud_prevented_usd'] / 1000,  # in thousands
        business['roi_pct']
    ]
    ax2_twin = ax2.twinx()
    
    bar1 = ax2.bar([0], [impact_values[0]], color='#27ae60', alpha=0.8, width=0.4)
    bar2 = ax2_twin.bar([1], [impact_values[1]], color='#e74c3c', alpha=0.8, width=0.4)
    
    ax2.set_ylabel('Amount ($K)', fontweight='bold', color='#27ae60')
    ax2_twin.set_ylabel('ROI (%)', fontweight='bold', color='#e74c3c')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(impact_labels)
    ax2.set_title('Annual Business Impact', fontweight='bold')
    ax2.text(0, impact_values[0], f'${impact_values[0]:.0f}K', ha='center', va='bottom', fontweight='bold')
    ax2_twin.text(1, impact_values[1], f'{impact_values[1]:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Class Distribution
    ax3 = axes[0, 2]
    class_counts = pd.Series(y_test).value_counts()
    colors_pie = ['#3498db', '#e74c3c']
    wedges, texts, autotexts = ax3.pie(class_counts, labels=['Normal', 'Fraud'],
                                         autopct='%1.2f%%', colors=colors_pie,
                                         startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax3.set_title('Test Set Class Distribution', fontweight='bold')
    
    # 4. Transaction Amount Distribution
    ax4 = axes[1, 0]
    normal_amounts = X_test[y_test == 0]['Amount']
    fraud_amounts = X_test[y_test == 1]['Amount']
    
    ax4.hist(normal_amounts, bins=50, alpha=0.6, label='Normal', color='#3498db', range=(0, 500))
    ax4.hist(fraud_amounts, bins=50, alpha=0.6, label='Fraud', color='#e74c3c', range=(0, 500))
    ax4.set_xlabel('Transaction Amount ($)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Amount Distribution by Class', fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Feature Importance (Top 10)
    ax5 = axes[1, 1]
    feat_imp = pd.read_csv('models/feature_importance.csv').head(10)
    ax5.barh(feat_imp['feature'], feat_imp['importance'], color='#9b59b6', alpha=0.8)
    ax5.set_xlabel('SHAP Importance', fontweight='bold')
    ax5.set_title('Top 10 Most Important Features', fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)
    
    # 6. Summary Stats
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    KEY PERFORMANCE INDICATORS
    
    Model Metrics:
    • Precision: {metadata['ensemble_metrics']['precision']:.1%}
    • Recall: {metadata['ensemble_metrics']['recall']:.1%}
    • F1-Score: {metadata['ensemble_metrics']['f1']:.3f}
    
    Business Impact:
    • Frauds Detected: {business['fraud_detected']}
    • Value Saved: ${business['value_saved_usd']:,.2f}
    • Annual Prevention: ${business['annual_fraud_prevented_usd']:,.0f}
    • ROI: {business['roi_pct']:.0f}%
    
    Data Quality:
    • Test Samples: {len(X_test):,}
    • Features: {X_test.shape[1]}
    • Fraud Rate: {(y_test.sum()/len(y_test)*100):.2f}%
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')
    
    plt.tight_layout()
    plt.savefig('models/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n   ✅ Saved: models/monitoring_dashboard.png")
    
    # Create performance summary JSON
    performance_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_version": "1.0.0",
        "performance": {
            "precision": metadata['ensemble_metrics']['precision'],
            "recall": metadata['ensemble_metrics']['recall'],
            "f1_score": metadata['ensemble_metrics']['f1']
        },
        "business_impact": {
            "frauds_detected": business['fraud_detected'],
            "value_saved_usd": business['value_saved_usd'],
            "annual_projection_usd": business['annual_fraud_prevented_usd'],
            "roi_percent": business['roi_pct']
        },
        "data_stats": {
            "test_samples": int(len(X_test)),
            "fraud_cases": int(y_test.sum()),
            "fraud_rate_percent": float(y_test.sum() / len(y_test) * 100)
        }
    }
    
    with open('models/performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    print("   ✅ Saved: models/performance_summary.json")
    
    print("\n" + "="*60)
    print("✅ MONITORING DASHBOARD COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    create_monitoring_dashboard()