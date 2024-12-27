# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path = '/path_to_dataset/Attrition_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Drop columns with constant values
data.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Handle missing values by filling with mean for numerical columns
for col in ['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 
            'JobSatisfaction', 'WorkLifeBalance']:
    data[col].fillna(data[col].mean(), inplace=True)

# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split the data into features and target
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 3: Build a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = clf.predict(X_test)

# Print evaluation metrics
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 5: Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importances")
plt.show()
