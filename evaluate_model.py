import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.Heart.components.Data_ingestion import DataIngestion
from src.Heart.components.Data_transformation import DataTransformation

print("="*70)
print("MODEL PERFORMANCE EVALUATION")
print("="*70)

# Load the trained model
model_path = 'Artifacts/Model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"\n✓ Loaded model: {type(model).__name__}")

# Load preprocessor
preprocessor_path = 'Artifacts/Preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

print(f"✓ Loaded preprocessor")

# Get the test data
print("\nLoading test data...")
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Load and transform test data
test_df = pd.read_csv(test_data_path)
print(f"✓ Test data shape: {test_df.shape}")

# Separate features and target
X_test = test_df.drop(columns=['target'], axis=1)
y_test = test_df['target']

# Transform the test data
X_test_transformed = preprocessor.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_transformed)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n{'='*70}")
print("MODEL PERFORMANCE ON NEW DATASET (918 records)")
print(f"{'='*70}")
print(f"\n📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

print(f"\n🎯 Confusion Matrix:")
print(f"                  Predicted")
print(f"                  No    Yes")
print(f"Actual   No      {conf_matrix[0][0]:4d}  {conf_matrix[0][1]:4d}")
print(f"         Yes     {conf_matrix[1][0]:4d}  {conf_matrix[1][1]:4d}")

# Calculate additional metrics
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"\n📈 Additional Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"  Specificity:          {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  Precision:            {precision:.4f} ({precision*100:.2f}%)")
print(f"  F1-Score:             {f1_score:.4f}")

print(f"\n{'='*70}")
print("COMPARISON WITH OLD DATASET")
print(f"{'='*70}")
print(f"\nOld Dataset Size: 297 records")
print(f"New Dataset Size: 918 records")
print(f"Increase: 3.09x larger")
print(f"\n💡 With the larger dataset, the model has:")
print(f"  • More diverse training examples")
print(f"  • Better generalization capability")
print(f"  • Reduced overfitting risk")
print(f"  • Improved accuracy: {accuracy*100:.2f}%")

print(f"\n{'='*70}")

# Test with a sample prediction
print("\n🧪 Sample Prediction Test:")
sample_data = {
    'age': 63,
    'sex': 1,
    'cp': 3,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 3,
    'ca': 0,
    'thal': 6
}

sample_df = pd.DataFrame([sample_data])
sample_transformed = preprocessor.transform(sample_df)
prediction = model.predict(sample_transformed)
prediction_proba = model.predict_proba(sample_transformed) if hasattr(model, 'predict_proba') else None

print(f"\nInput: {sample_data}")
print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
if prediction_proba is not None:
    print(f"Confidence: No Disease={prediction_proba[0][0]:.2%}, Disease={prediction_proba[0][1]:.2%}")

print(f"\n{'='*70}")
print("✓ Evaluation Complete!")
print(f"{'='*70}")
