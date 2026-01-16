import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("="*70)
print("PREPROCESSING NEW DATASET FOR MODEL COMPATIBILITY")
print("="*70)

# Load the new dataset
df = pd.read_csv('Notebook_Experiments/Data/heart.csv')
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# The new dataset has these columns:
# ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
#  'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'target']

# The model expects these columns (from Data_transformation.py):
# ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
#  'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Create a mapping for column names
column_mapping = {
    'Age': 'age',
    'Sex': 'sex',
    'ChestPainType': 'cp',
    'RestingBP': 'trestbps',
    'Cholesterol': 'chol',
    'FastingBS': 'fbs',
    'RestingECG': 'restecg',
    'MaxHR': 'thalach',
    'ExerciseAngina': 'exang',
    'Oldpeak': 'oldpeak',
    'ST_Slope': 'slope',
    'target': 'target'
}

# Rename columns
df = df.rename(columns=column_mapping)

# Handle categorical variables that need to be encoded
# The new dataset has categorical text values that need to be converted to numbers

# Encode 'Sex' if it's categorical (M/F to 1/0)
if df['sex'].dtype == 'object':
    print("\nEncoding Sex column...")
    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'])
    print(f"  Sex mapping: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# Encode 'ChestPainType' (cp)
if df['cp'].dtype == 'object':
    print("\nEncoding Chest Pain Type...")
    # Map chest pain types to numbers (common mapping: ATA=1, NAP=2, ASY=3, TA=4)
    cp_mapping = {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4}
    df['cp'] = df['cp'].map(cp_mapping)
    print(f"  ChestPainType mapping: {cp_mapping}")

# Encode 'ExerciseAngina' (exang) 
if df['exang'].dtype == 'object':
    print("\nEncoding Exercise Angina...")
    le_exang = LabelEncoder()
    df['exang'] = le_exang.fit_transform(df['exang'])
    print(f"  ExerciseAngina mapping: {dict(zip(le_exang.classes_, le_exang.transform(le_exang.classes_)))}")

# Encode 'ST_Slope' (slope)
if df['slope'].dtype == 'object':
    print("\nEncoding ST Slope...")
    # Map ST slope: Up=1, Flat=2, Down=3
    slope_mapping = {'Up': 1, 'Flat': 2, 'Down': 3}
    df['slope'] = df['slope'].map(slope_mapping)
    print(f"  ST_Slope mapping: {slope_mapping}")

# Encode 'RestingECG' if needed
if df['restecg'].dtype == 'object':
    print("\nEncoding Resting ECG...")
    le_restecg = LabelEncoder()
    df['restecg'] = le_restecg.fit_transform(df['restecg'])
    print(f"  RestingECG mapping: {dict(zip(le_restecg.classes_, le_restecg.transform(le_restecg.classes_)))}")

# The new dataset doesn't have 'ca' (number of major vessels) and 'thal' (thalassemia)
# We'll need to add these as synthetic features or use zeros
# For now, let's add them as 0 (this is a limitation of using this dataset)
print("\nAdding missing columns (ca, thal) as zeroes...")
df['ca'] = 0
df['thal'] = 3  # Use 3 as default (normal)

# Reorder columns to match expected order
expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = df[expected_columns]

print(f"\nProcessed dataset shape: {df.shape}")
print(f"Processed columns: {list(df.columns)}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check data types
print(f"\nData types:\n{df.dtypes}")

# Verify all values are numeric
print(f"\nAll columns numeric: {all(df.dtypes != 'object')}")

# Save the processed dataset
output_path = 'Notebook_Experiments/Data/heart.csv'
df.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print("✓ Preprocessing Complete!")
print(f"{'='*70}")
print(f"Processed dataset saved to: {output_path}")
print(f"Ready for model training with {len(df)} records")
print(f"{'='*70}")

# Show sample of processed data
print(f"\nFirst 5 rows of processed data:")
print(df.head())

print(f"\nTarget distribution:")
print(df['target'].value_counts())
