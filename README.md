# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the weather station dataset.

2.Preprocess the dataset by removing unnecessary columns and handling missing values.

3.Separate the dataset into input features and target variables such as temperature, PM2.5, and energy (tsr).

4.Split the dataset into training and testing data using train_test_split().

5.Create the Random Forest Regressor model using RandomForestRegressor().

6.Train the model using training data.

7.Predict the output values for testing data.

8.Evaluate the model performance using MAE and R2 Score.

9.Plot the feature importance graph for visualization.

10.Display the predicted results and stop the program.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Piruthiviraj G
RegisterNumber:  212225040299
*/
```
```python
# Random Forest Prediction using Weather Station Data

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")

# Step 3: Display First 5 Rows
print(df.head())

# Step 4: Remove Unnecessary Columns
df = df.drop(columns=['time', 'wind_direction', 'bat'])

# Step 5: Handle Missing Values
df = df.fillna(df.mean(numeric_only=True))

# ---------------------------------------------------
# TEMPERATURE PREDICTION
# ---------------------------------------------------

# Features and Target
X_temp = df.drop(columns=['tem'])
y_temp = df['tem']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

# Create Model
temp_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train Model
temp_model.fit(X_train, y_train)

# Predict
y_pred = temp_model.predict(X_test)

# Evaluation
print("\nTemperature Prediction")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------------------------------------------
# PM2.5 PREDICTION
# ---------------------------------------------------

X_pm = df.drop(columns=['pm2_5'])
y_pm = df['pm2_5']

X_train, X_test, y_train, y_test = train_test_split(
    X_pm, y_pm, test_size=0.2, random_state=42
)

pm_model = RandomForestRegressor(n_estimators=100, random_state=42)

pm_model.fit(X_train, y_train)

y_pred = pm_model.predict(X_test)

print("\nPM2.5 Prediction")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------------------------------------------
# ENERGY (TSR) PREDICTION
# ---------------------------------------------------

X_energy = df.drop(columns=['tsr'])
y_energy = df['tsr']

X_train, X_test, y_train, y_test = train_test_split(
    X_energy, y_energy, test_size=0.2, random_state=42
)

energy_model = RandomForestRegressor(n_estimators=100, random_state=42)

energy_model.fit(X_train, y_train)

y_pred = energy_model.predict(X_test)

print("\nEnergy Prediction")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------------------------------------------
# FEATURE IMPORTANCE GRAPH
# ---------------------------------------------------

importance = temp_model.feature_importances_
features = X_temp.columns

plt.figure(figsize=(10, 6))
plt.bar(features, importance)

plt.title("Feature Importance for Temperature Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.show()

```

## Output:
<img width="559" height="513" alt="image" src="https://github.com/user-attachments/assets/dca2f60a-66d0-47d2-a953-161d6b8f8d43" />


## Result:
Thus, the program to predict daily temperature, PM2.5 pollution level, and energy using the Random Forest Algorithm was implemented successfully.
