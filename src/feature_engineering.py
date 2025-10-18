import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_advanced_features(df):
    """
    Create powerful fraud detection features
    """
    print("🔧 Starting feature engineering...")
    
    df = df.copy()
    
    # 1. TIME-BASED FEATURES (fraudsters operate at unusual hours)
    df['hour'] = (df['Time'] / 3600) % 24  # Hour of day
    df['day'] = (df['Time'] / 86400).astype(int)  # Day number
    
    # Cyclical encoding for hour (fraud patterns repeat daily)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    print("   ✅ Time-based features created")
    
    # 2. AMOUNT-BASED FEATURES
    df['amount_log'] = np.log1p(df['Amount'])  # Log transform (handles skewness)
    
    # Amount percentiles (is this unusually high?)
    amount_percentiles = df['Amount'].quantile([0.25, 0.50, 0.75, 0.90, 0.95])
    df['amount_percentile'] = df['Amount'].apply(
        lambda x: sum(x > amount_percentiles)
    )
    
    # Decimal patterns (fraudsters often use round numbers)
    df['amount_decimal'] = df['Amount'] % 1
    df['is_round_amount'] = (df['Amount'] % 1 == 0).astype(int)
    
    print("   ✅ Amount-based features created")
    
    # 3. STATISTICAL FEATURES FROM V1-V28
    v_columns = [f'V{i}' for i in range(1, 29)]
    
    # Aggregate statistics
    df['v_mean'] = df[v_columns].mean(axis=1)
    df['v_std'] = df[v_columns].std(axis=1)
    df['v_min'] = df[v_columns].min(axis=1)
    df['v_max'] = df[v_columns].max(axis=1)
    df['v_range'] = df['v_max'] - df['v_min']
    
    # Count of extreme values (outliers often indicate fraud)
    df['v_extreme_count'] = (np.abs(df[v_columns]) > 3).sum(axis=1)
    
    print("   ✅ Statistical aggregation features created")
    
    # 4. INTERACTION FEATURES (combinations matter!)
    # These are domain-specific - some V features are more important together
    df['v1_v2_interaction'] = df['V1'] * df['V2']
    df['v4_amount_interaction'] = df['V4'] * df['amount_log']
    
    print("   ✅ Interaction features created")
    
    print(f"\n📊 Feature engineering complete!")
    print(f"   Original features: 31")
    print(f"   New features created: {len(df.columns) - 31}")
    print(f"   Total features: {len(df.columns)}")
    
    return df

def prepare_data_for_modeling(df):
    """
    Prepare final dataset for model training
    """
    print("\n🔧 Preparing data for modeling...")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"   ✅ Removed {initial_rows - len(df)} duplicate rows")
    
    # Separate features and target
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # Split data (80% train, 20% test)
    # Stratify to maintain fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n📊 Data split complete:")
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   - Normal: {(y_train==0).sum():,}")
    print(f"   - Fraud: {(y_train==1).sum():,}")
    print(f"   Test set: {len(X_test):,} samples")
    print(f"   - Normal: {(y_test==0).sum():,}")
    print(f"   - Fraud: {(y_test==1).sum():,}")
    
    # Scale features (critical for many ML algorithms)
    # We scale AFTER split to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"   ✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('data/creditcard.csv')
    
    # Create features
    df_engineered = create_advanced_features(df)
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_modeling(df_engineered)
    
    # Save processed data
    print("\n💾 Saving processed data...")
    
    # Save train/test splits
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    # Save scaler for production use
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"   ✅ Training data saved: data/X_train.csv")
    print(f"   ✅ Test data saved: data/X_test.csv")
    print(f"   ✅ Scaler saved: models/scaler.pkl")
    print(f"\n{'='*60}")
    print(f"✅ FEATURE ENGINEERING COMPLETE!")
    print(f"{'='*60}")