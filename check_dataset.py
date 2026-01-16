import pandas as pd

# Load and analyze current dataset
df = pd.read_csv('Notebook_Experiments/Data/heart.csv')

print(f'Current dataset shape: {df.shape}')
print(f'\nColumns: {list(df.columns)}')
print(f'\nFirst few rows:\n{df.head()}')
print(f'\nDataset info:')
df.info()
print(f'\nTarget distribution:\n{df["target"].value_counts() if "target" in df.columns else "No target column"}')
print(f'\nMissing values:\n{df.isnull().sum()}')
print(f'\nBasic statistics:\n{df.describe()}')
