import pandas as pd
import requests
from io import StringIO
import os

print("=" * 70)
print("ENHANCED HEART DISEASE DATASET DOWNLOADER")
print("Upgrading from 297 to 1,190+ records")
print("=" * 70)

# Multiple sources for the comprehensive heart failure prediction dataset
# This dataset combines 5 different heart disease datasets
sources = [
    {
        "name": "Heart Failure Dataset - xpy-10 Mirror (Primary)",
        "url": "https://raw.githubusercontent.com/xpy-10/DataSet/main/heart.csv",
        "has_header": True,
        "priority": 1
    },
    {
        "name": "Heart Disease Dataset - UCI Combined",
        "url": "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI/master/heart.csv",
        "has_header": True,
        "priority": 2
    },
    {
        "name": "Heart Dataset - Alternative Source",
        "url": "https://raw.githubusercontent.com/Sid-darthvader/Heart-Disease-Prediction/master/heart.csv",
        "has_header": True,
        "priority": 3
    }
]

# Sort by priority
sources.sort(key=lambda x: x['priority'])

# Expected columns for the comprehensive dataset
expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                   'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                   'Oldpeak', 'ST_Slope', 'HeartDisease']

# Alternative column names that might be used
alt_column_mappings = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'ChestPainType',
    'trestbps': 'RestingBP',
    'chol': 'Cholesterol',
    'fbs': 'FastingBS',
    'restecg': 'RestingECG',
    'thalach': 'MaxHR',
    'exang': 'ExerciseAngina',
    'oldpeak': 'Oldpeak',
    'slope': 'ST_Slope',
    'target': 'HeartDisease',
    'heartdisease': 'HeartDisease'
}

dataset_downloaded = False

for source in sources:
    try:
        print(f"\n{'='*70}")
        print(f"ATTEMPTING: {source['name']}")
        print(f"URL: {source['url']}")
        print(f"{'='*70}")
        
        response = requests.get(source['url'], timeout=15)
        response.raise_for_status()
        
        # Load the data
        df = pd.read_csv(StringIO(response.text))
        
        print(f"\n✓ Successfully downloaded!")
        print(f"  Initial shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Normalize column names if needed
        df.columns = df.columns.str.strip()
        
        # Map alternative column names to expected names
        df = df.rename(columns={col.lower(): alt_column_mappings.get(col.lower(), col) 
                               for col in df.columns})
        
        # Clean the data
        print(f"\nCleaning data...")
        initial_rows = len(df)
        
        # Replace '?' or other placeholders with NaN
        df = df.replace('?', pd.NA)
        df = df.replace('', pd.NA)
        
        # For numeric columns, convert to numeric
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna()
        
        print(f"  Rows before cleaning: {initial_rows}")
        print(f"  Rows after cleaning: {len(df)}")
        print(f"  Rows removed: {initial_rows - len(df)}")
        
        # Validate we have enough data
        if len(df) < 500:
            print(f"  ⚠ WARNING: Dataset too small ({len(df)} rows), trying next source...")
            continue
        
        # Ensure target column exists and is binary
        target_col = None
        for col in ['HeartDisease', 'target', 'Target']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print(f"  ✗ ERROR: No target column found!")
            continue
        
        # Rename target column to 'target' for consistency
        if target_col != 'target':
            df = df.rename(columns={target_col: 'target'})
        
        # Ensure target is binary (0 or 1)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Create output directory if it doesn't exist
        output_dir = "Notebook_Experiments/Data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the original dataset as backup
        backup_path = os.path.join(output_dir, "heart_old_backup.csv")
        if os.path.exists(os.path.join(output_dir, "heart.csv")) and not os.path.exists(backup_path):
            import shutil
            print(f"\nBacking up old dataset to: {backup_path}")
            shutil.copy(os.path.join(output_dir, "heart.csv"), backup_path)
        
        # Save the new dataset
        output_path = os.path.join(output_dir, "heart.csv")
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"✓✓✓ SUCCESS! ✓✓✓")
        print(f"{'='*70}")
        print(f"\nDataset Details:")
        print(f"  Total Records: {len(df)}")
        print(f"  Total Features: {len(df.columns)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nTarget Distribution:")
        print(f"  No Heart Disease (0): {(df['target'] == 0).sum()} samples ({(df['target'] == 0).sum()/len(df)*100:.1f}%)")
        print(f"  Heart Disease (1): {(df['target'] == 1).sum()} samples ({(df['target'] == 1).sum()/len(df)*100:.1f}%)")
        print(f"\nData Quality:")
        print(f"  Missing Values: {df.isnull().sum().sum()}")
        print(f"  Duplicate Rows: {df.duplicated().sum()}")
        print(f"\nAge Statistics:")
        if 'Age' in df.columns or 'age' in df.columns:
            age_col = 'Age' if 'Age' in df.columns else 'age'
            print(f"  Min Age: {df[age_col].min():.0f}")
            print(f"  Max Age: {df[age_col].max():.0f}")
            print(f"  Mean Age: {df[age_col].mean():.1f}")
        
        print(f"\nDataset saved to: {output_path}")
        print(f"Old dataset backed up to: {backup_path}")
        print(f"\n{'='*70}")
        print("UPGRADE COMPLETE!")
        print(f"Dataset size increased from 297 to {len(df)} records")
        print(f"That's a {len(df)/297:.1f}x increase!")
        print(f"{'='*70}")
        
        dataset_downloaded = True
        break
        
    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        print(f"Trying next source...")
        continue

if not dataset_downloaded:
    print("\n" + "="*70)
    print("❌ ERROR: All sources failed!")
    print("="*70)
    print("\nPlease check your internet connection and try again.")
    print("You can also manually download the dataset from:")
    print("https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction")
    exit(1)
