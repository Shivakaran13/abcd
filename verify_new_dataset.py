import pandas as pd

# Load the new dataset
df_new = pd.read_csv('Notebook_Experiments/Data/heart.csv')

print("="*70)
print("NEW DATASET VERIFICATION")
print("="*70)
print(f"\nDataset Shape: {df_new.shape}")
print(f"Rows: {df_new.shape[0]}")
print(f"Columns: {df_new.shape[1]}")

print(f"\nColumns: {list(df_new.columns)}")

print(f"\nFirst 5 rows:")
print(df_new.head())

print(f"\nData Types:")
print(df_new.dtypes)

print(f"\nMissing Values:")
print(df_new.isnull().sum())

print(f"\nTarget Distribution:")
if 'target' in df_new.columns:
    print(df_new['target'].value_counts())
    print(f"\nClass Balance:")
    print(f"  Class 0: {(df_new['target']==0).sum()} ({(df_new['target']==0).sum()/len(df_new)*100:.1f}%)")
    print(f"  Class 1: {(df_new['target']==1).sum()} ({(df_new['target']==1).sum()/len(df_new)*100:.1f}%)")

print(f"\nBasic Statistics:")
print(df_new.describe())

# Check if backup exists
import os
if os.path.exists('Notebook_Experiments/Data/heart_old_backup.csv'):
    df_old = pd.read_csv('Notebook_Experiments/Data/heart_old_backup.csv')
    print(f"\n{'='*70}")
    print("COMPARISON WITH OLD DATASET")
    print(f"{'='*70}")
    print(f"Old dataset rows: {len(df_old)}")
    print(f"New dataset rows: {len(df_new)}")
    print(f"Increase: {len(df_new) - len(df_old)} rows ({(len(df_new)/len(df_old) - 1)*100:.1f}% more)")
    print(f"Multiplier: {len(df_new)/len(df_old):.2f}x")
