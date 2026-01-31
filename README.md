<img width="718" height="583" alt="Screenshot 2026-01-31 142411" src="https://github.com/user-attachments/assets/c7636e91-2fdd-4ba6-bc1a-006007704f5e" /># Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
   

## Program:
```
# Logistic Regression for Student Placement Prediction

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 2️⃣ Load Dataset
data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

# 3️⃣ Drop Unnecessary Columns
data = data.drop(["sl_no", "salary"], axis=1)

# 4️⃣ Convert Target Variable (status) to Binary
# Placed = 1, Not Placed = 0
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

# 5️⃣ Separate Features and Target
X = data.drop("status", axis=1)
y = data["status"]

# 6️⃣ One-Hot Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

# 7️⃣ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9️⃣ Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔟 Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 1️⃣1️⃣ Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mukesh M
RegisterNumber:  212225240093
*/
```

## Output:
<img width="1056" height="756" alt="Screenshot 2026-01-31 142459" src="https://github.com/user-attachments/assets/29494510-9f1f-46ac-8936-722d921823d3" />
<img width="718" height="583" alt="Screenshot 2026-01-31 142411" src="https://github.com/user-attachments/assets/f428a89a-8e9e-42d4-b8ef-7955bfb67dd0" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
