import pandas as pd
import requests
from io import StringIO

# Try multiple high-quality sources for the UCI heart disease dataset
sources = [
    {
        "name": "UCI ML Repository - Cleveland",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "has_header": False
    },
    {
        "name": "Kaggle mirror - Heart CSV",
        "url": "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI/master/heart.csv",
        "has_header": True
    }
]

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

for source in sources:
    try:
        print(f"\n=== Trying: {source['name']} ===")
        print(f"URL: {source['url']}")
        
        response = requests.get(source['url'], timeout=10)
        response.raise_for_status()
        
        if source['has_header']:
            df = pd.read_csv(StringIO(response.text))
        else:
            df = pd.read_csv(StringIO(response.text), header=None, names=column_names)
        
        # Clean the data
        df = df.replace('?', pd.NA).dropna()
        
        # Ensure target is binary (0 or 1)
        if 'target' in df.columns:
            # If target has multiple classes (0, 1, 2, 3, 4), convert to binary
            df['target'] = (df['target'] > 0).astype(int)
        
        # Save the dataset
        df.to_csv("Notebook_Experiments/Data/heart.csv", index=False)
        
        print(f"✓ SUCCESS!")
        print(f"  Downloaded {len(df)} rows with {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Target distribution:")
        print(f"    - No disease (0): {(df['target'] == 0).sum()} samples")
        print(f"    - Disease (1): {(df['target'] == 1).sum()} samples")
        print(f"\nDataset saved to: Notebook_Experiments/Data/heart.csv")
        break
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        continue
else:
    print("\n❌ All sources failed. Dataset not updated.")
