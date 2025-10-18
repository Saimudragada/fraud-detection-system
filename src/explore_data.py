import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading dataset...")
df = pd.read_csv('data/creditcard.csv')

print(f"\n{'='*60}")
print(f"✅ DATASET LOADED SUCCESSFULLY!")
print(f"{'='*60}")

print(f"\n📊 BASIC INFORMATION:")
print(f"   Dataset shape: {df.shape}")
print(f"   Total transactions: {len(df):,}")
print(f"   Total features: {len(df.columns)}")

print(f"\n📋 COLUMNS:")
print(f"   {df.columns.tolist()}")

print(f"\n🔍 FIRST 5 ROWS:")
print(df.head())

print(f"\n📈 DATASET INFO:")
df.info()

print(f"\n{'='*60}")
print(f"🚨 FRAUD STATISTICS")
print(f"{'='*60}")
fraud_count = df['Class'].sum()
normal_count = (df['Class']==0).sum()
fraud_pct = df['Class'].mean()*100

print(f"   Normal transactions: {normal_count:,} ({100-fraud_pct:.2f}%)")
print(f"   Fraud transactions:  {fraud_count:,} ({fraud_pct:.4f}%)")
print(f"   Class imbalance ratio: {normal_count}:{fraud_count} (~{normal_count//fraud_count}:1)")

print(f"\n💰 AMOUNT STATISTICS:")
print(df.groupby('Class')['Amount'].describe())

print(f"\n⏰ TIME STATISTICS:")
print(f"   Time range: {df['Time'].min():.0f}s to {df['Time'].max():.0f}s")
print(f"   Duration: ~{df['Time'].max()/3600:.1f} hours")

print(f"\n✅ DATA QUALITY:")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Duplicate rows: {df.duplicated().sum()}")

print(f"\n{'='*60}")
print(f"✅ EXPLORATION COMPLETE!")
print(f"{'='*60}")