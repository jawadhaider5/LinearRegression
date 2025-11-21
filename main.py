# PROBLEM
# Predict median house value for California Districts using numerical and categorical features.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

housing = fetch_california_housing(as_frame=True)
housing_data = housing.frame
print(housing_data.head())
print(housing_data.isnull().values.any())

X = housing_data.drop(columns=["MedHouseVal"])  # features
y = housing_data["MedHouseVal"]                 # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nLinear Regression:")
print(f"R² = {r2:.3f} | RMSE = {rmse:.3f}")

corr_matrix = housing_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=.5)
plt.title("Correlation Heatmap - California Housing Dataset")
plt.show()