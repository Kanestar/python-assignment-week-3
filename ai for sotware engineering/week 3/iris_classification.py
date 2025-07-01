# Task 1: Classical ML with Scikit-learn
# Dataset: Iris Species Dataset
# Goal: Preprocess data, train a decision tree classifier, evaluate using accuracy, precision, and recall.

# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("Starting Iris Species Classification Task...")

# 1. Load the Iris Dataset
# The Iris dataset is a classic and is included in scikit-learn
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (species)
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset loaded: {len(X)} samples, {len(feature_names)} features, {len(target_names)} species.")

# Convert to DataFrame for easier manipulation and to check for missing values
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

# 2. Preprocess the data

# Handle missing values: The Iris dataset is known to be clean and typically has no missing values.
# However, it's good practice to include a check.
print("Checking for missing values...")
if df.isnull().sum().sum() > 0:
    print("Missing values found. Handling them...")
    # For this dataset, we can simply drop rows with missing values if any appear
    df.dropna(inplace=True)
    print("Missing values handled by dropping rows.")
else:
    print("No missing values found.")

# Encode labels: The 'species' target variable is already numerical (0, 1, 2).
# If it were string labels (e.g., 'setosa', 'versicolor', 'virginica'),
# we would use LabelEncoder.
# Let's demonstrate LabelEncoder for completeness if we were to work with string labels.
# For this dataset, we'll use the existing numerical target 'y'.
print("Encoding labels (if necessary)...")
le = LabelEncoder()
# Fit and transform the numerical labels to themselves (no change, just for demonstration)
# This step is technically not needed for iris.target as it's already numerical
# But if 'y' contained strings like ['setosa', 'versicolor', 'virginica'], this would be essential.
# y_encoded = le.fit_transform(y) # This would be used if y was string labels
# print(f"Labels encoded to: {le.classes_}")
# print(f"First 5 original labels: {y[:5]}")
# print(f"First 5 encoded labels: {y_encoded[:5]}")

# 3. Split the data into training and testing sets
# We use a 80/20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data split into training (80%) and testing (20%).")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 4. Train a Decision Tree Classifier
print("Training Decision Tree Classifier...")
clf = DecisionTreeClassifier(random_state=42) # Initialize the classifier
clf.fit(X_train, y_train) # Train the model on the training data
print("Decision Tree Classifier trained successfully.")

# 5. Make predictions on the test set
print("Making predictions on the test set...")
y_pred = clf.predict(X_test)
print("Predictions made.")

# 6. Evaluate the model
print("Evaluating the model...")

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision Score (weighted to account for potential imbalance, though Iris is balanced)
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision (weighted): {precision:.4f}")

# Recall Score (weighted)
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall (weighted): {recall:.4f}")

# Classification Report (provides precision, recall, f1-score for each class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Iris Species Classification Task completed.") 