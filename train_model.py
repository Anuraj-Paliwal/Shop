import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

# Load the credit card dataset
print("📊 Loading credit card dataset...")
df = pd.read_csv("creditcard.csv")

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Genuine cases: {(df['Class'] == 0).sum()}")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🔧 Training Random Forest Classifier...")
# Train Random Forest (handles imbalanced data well)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight='balanced',  # Handle imbalanced data
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate model
print("\n✅ Model Training Complete!")
y_pred = model.predict(X_test)

print("\n📈 Model Performance:")
print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
print("\n💾 Saving model...")
joblib.dump(model, 'fraud_model.pkl')

# Save feature names for later use
feature_names = list(X.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Save scaler (if we were using one, but RF doesn't need it)
# For production, you might want to normalize Amount and Time
print("✅ Model saved as 'fraud_model.pkl'")
print("✅ Feature names saved as 'feature_names.json'")

# Test prediction
print("\n🧪 Testing prediction on a sample transaction...")
sample = X_test.iloc[0:1]
prediction = model.predict(sample)
probability = model.predict_proba(sample)

print(f"Prediction: {'FRAUD' if prediction[0] == 1 else 'GENUINE'}")
print(f"Fraud Probability: {probability[0][1]:.2%}")