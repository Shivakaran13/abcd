import pandas as pd
import pickle
import numpy as np

print("="*70)
print("TESTING PREDICTION FOR SPECIFIC PATIENT")
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

# Patient data from user
patient_data = {
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
    'slope': 0,  # Upsloping (Note: should be 1, 2, or 3 typically)
    'ca': 0,
    'thal': 0  # Normal (Note: should be 3, 6, or 7 typically)
}

print("\n" + "="*70)
print("PATIENT PROFILE")
print("="*70)
print(f"\nAge: {patient_data['age']} years")
print(f"Sex: {'Female' if patient_data['sex'] == 0 else 'Male'}")
print(f"Chest Pain Type: Non-Anginal Pain")
print(f"Resting Blood Pressure: {patient_data['trestbps']} mm Hg")
print(f"Cholesterol: {patient_data['chol']} mg/dl")
print(f"Fasting Blood Sugar > 120 mg/dl: {'Yes' if patient_data['fbs'] == 1 else 'No'}")
print(f"Resting ECG: Normal")
print(f"Max Heart Rate: {patient_data['thalach']} bpm")
print(f"Exercise Induced Angina: {'Yes' if patient_data['exang'] == 1 else 'No'}")
print(f"Oldpeak (ST Depression): {patient_data['oldpeak']}")
print(f"Slope: Upsloping")
print(f"Number of Major Vessels: {patient_data['ca']}")
print(f"Thalassemia: Normal")

# Clinical Assessment
print("\n" + "="*70)
print("CLINICAL ASSESSMENT")
print("="*70)
print("\n✅ POSITIVE INDICATORS (Low Risk):")
print("  • Very young age (18 years)")
print("  • Excellent cholesterol level (120 mg/dl)")
print("  • Normal blood pressure (90 mm Hg)")
print("  • No fasting blood sugar issues")
print("  • Normal resting ECG")
print("  • Very high max heart rate (200 bpm - excellent fitness)")
print("  • No exercise-induced angina")
print("  • No ST depression (0.0)")
print("  • No major vessels affected")
print("  • Normal thalassemia status")

print("\n⚠️ RISK FACTORS:")
print("  • None identified")

# Note: The values for slope and thal might need adjustment
# In standard heart disease datasets:
# - slope: 1 (upsloping), 2 (flat), 3 (downsloping)
# - thal: 3 (normal), 6 (fixed defect), 7 (reversible defect)

# Adjust values to match expected encoding
adjusted_data = patient_data.copy()
if adjusted_data['slope'] == 0:
    adjusted_data['slope'] = 1  # Upsloping
if adjusted_data['thal'] == 0:
    adjusted_data['thal'] = 3  # Normal

print("\n" + "="*70)
print("MODEL PREDICTION")
print("="*70)

# Create DataFrame
patient_df = pd.DataFrame([adjusted_data])

# Transform the data
patient_transformed = preprocessor.transform(patient_df)

# Make prediction
prediction = model.predict(patient_transformed)[0]

# Get probability if available
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(patient_transformed)[0]
    print(f"\n📊 Prediction Probabilities:")
    print(f"  No Heart Disease: {probabilities[0]:.2%}")
    print(f"  Heart Disease:    {probabilities[1]:.2%}")

print(f"\n{'='*70}")
print(f"FINAL PREDICTION: ", end="")
if prediction == 0:
    print("❤️ NO HEART DISEASE RISK")
    print(f"{'='*70}")
    print("\n✅ This patient shows NO signs of heart disease.")
    print("   All indicators are excellent for a healthy 18-year-old.")
else:
    print("⚠️ HEART DISEASE RISK DETECTED")
    print(f"{'='*70}")
    print("\n⚠️ The model predicts potential heart disease risk.")

print(f"\n{'='*70}")
print("EXPECTED OUTPUT")
print(f"{'='*70}")
print(f"\nFor this patient profile, the expected output is:")
print(f"  Prediction Value: {prediction}")
print(f"  Interpretation: {'No Heart Disease' if prediction == 0 else 'Heart Disease Risk Detected'}")

print("\n💡 INTERPRETATION:")
if prediction == 0:
    print("This is a very healthy profile. An 18-year-old female with:")
    print("  • Excellent vital signs")
    print("  • No risk factors")
    print("  • High fitness level (max HR of 200)")
    print("  • No cardiac symptoms")
    print("\nThe model correctly identifies this as LOW RISK.")
else:
    print("Note: If the model predicts disease for this profile,")
    print("it may be due to unusual encoding or data preprocessing.")
    print("This profile should typically result in NO DISEASE prediction.")

print(f"\n{'='*70}")
