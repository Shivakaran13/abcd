import pandas as pd
import pickle
import numpy as np

print("="*70)
print("INVESTIGATING UNEXPECTED PREDICTION")
print("="*70)

# Load the trained model
model_path = 'Artifacts/Model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
preprocessor_path = 'Artifacts/Preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

print(f"\n✓ Model loaded: {type(model).__name__}")

# EXACT values as user provided
patient_data_original = {
    'age': 18,
    'sex': 0,  # Female
    'cp': 2,   # Non-Anginal Pain
    'trestbps': 90,
    'chol': 120,
    'fbs': 0,  # No
    'restecg': 0,  # Normal
    'thalach': 200,
    'exang': 0,  # No
    'oldpeak': 0.0,
    'slope': 0,  # Upsloping - THIS MIGHT BE THE ISSUE!
    'ca': 0,
    'thal': 0  # Normal - THIS MIGHT BE THE ISSUE!
}

print("\n" + "="*70)
print("TEST 1: EXACT VALUES AS PROVIDED (slope=0, thal=0)")
print("="*70)

patient_df_original = pd.DataFrame([patient_data_original])
print(f"\nInput data:\n{patient_df_original}")

try:
    patient_transformed = preprocessor.transform(patient_df_original)
    prediction = model.predict(patient_transformed)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(patient_transformed)[0]
        print(f"\nPrediction Probabilities:")
        print(f"  No Heart Disease: {probabilities[0]:.2%}")
        print(f"  Heart Disease:    {probabilities[1]:.2%}")
    
    print(f"\n{'='*70}")
    print(f"PREDICTION: {prediction} ({'Heart Disease' if prediction == 1 else 'No Heart Disease'})")
    print(f"{'='*70}")
except Exception as e:
    print(f"\nError: {e}")

# Now test with corrected values
print("\n\n" + "="*70)
print("TEST 2: CORRECTED VALUES (slope=1, thal=3)")
print("="*70)
print("Note: In most heart disease datasets:")
print("  - slope: 1=upsloping, 2=flat, 3=downsloping")
print("  - thal: 3=normal, 6=fixed defect, 7=reversible defect")

patient_data_corrected = patient_data_original.copy()
patient_data_corrected['slope'] = 1  # Proper encoding for upsloping
patient_data_corrected['thal'] = 3   # Proper encoding for normal

patient_df_corrected = pd.DataFrame([patient_data_corrected])
print(f"\nCorrected input data:\n{patient_df_corrected}")

try:
    patient_transformed_corrected = preprocessor.transform(patient_df_corrected)
    prediction_corrected = model.predict(patient_transformed_corrected)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities_corrected = model.predict_proba(patient_transformed_corrected)[0]
        print(f"\nPrediction Probabilities:")
        print(f"  No Heart Disease: {probabilities_corrected[0]:.2%}")
        print(f"  Heart Disease:    {probabilities_corrected[1]:.2%}")
    
    print(f"\n{'='*70}")
    print(f"PREDICTION: {prediction_corrected} ({'Heart Disease' if prediction_corrected == 1 else 'No Heart Disease'})")
    print(f"{'='*70}")
except Exception as e:
    print(f"\nError: {e}")

# Check the training data to understand the encoding
print("\n\n" + "="*70)
print("CHECKING TRAINING DATA ENCODING")
print("="*70)

df = pd.read_csv('Notebook_Experiments/Data/heart.csv')
print(f"\nDataset slope values:")
print(df['slope'].value_counts().sort_index())
print(f"\nDataset thal values:")
print(df['thal'].value_counts().sort_index())

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nThe issue is likely that slope=0 and thal=0 are INVALID values")
print("in the standard heart disease dataset encoding.")
print("\nThe model was trained on:")
print("  - slope: 1, 2, or 3")
print("  - thal: 3, 6, or 7")
print("\nUsing 0 for these values creates out-of-distribution data")
print("that the model hasn't seen during training, leading to")
print("unexpected predictions.")
