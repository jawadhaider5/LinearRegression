# LinearRegression

*** California Housing — Linear Regression Project ***

Predicting median house values in California using a baseline Linear Regression model with standardized features.
This is my first machine learning project for GitHub, demonstrating a full regression workflow:

  data loading → preprocessing → modeling → evaluation → visualization


Project Overview

The goal of this project is to build a regression model that predicts "MedHouseVal"" (median house price) using the California Housing dataset provided by sklearn.
This dataset includes features such as:

* Median income
* Average rooms
* Average bedrooms
* Population
* Households
* Latitude & Longitude

Objectives

1. Load and explore the dataset
2. Prepare the data for modeling
3. Train a baseline linear regression model using a Pipeline
4. Standardize features for improved stability
5. Evaluate model performance using R² and RMSE
6. Visualize correlations between features and target

Dataset

The dataset is obtained using:

```python
from sklearn.datasets import fetch_california_housing
```

It contains 20,640 samples and 8 numerical features, plus the target variable:

MedHouseVal → Median house value (in $100,000 units)
No missing values were found in the dataset.

Methodology

1. Data Splitting

* Features (X) = all columns except `MedHouseVal`
* Target (y) = `MedHouseVal`
* Train-test split: **80% train / 20% test**

2. Preprocessing

Linear regression benefits from feature scaling.

3. Model Training

* Fit on training data
* Predict on test data

4. Evaluation Metrics

Two regression metrics were computed:

* R² Score
* RMSE (Root Mean Squared Error)

These measure how well the model explains variance and how far predictions are from true values.


Results

Model: Linear Regression (with StandardScaler)

| Metric       | Score     |
| ------------ | --------- |
| R² Score | 0.594 |
| RMSE     | 0.727 |


Interpretation

* The model explains 57.6% of the variance** in house prices.
* Average prediction error is $74,600, since the target is in units of $100,000.
* These numbers align with typical baseline performance for this dataset.
* Strongest predictor (from correlation analysis): Median Income (MedInc)
* Features like Latitude and Longitude also show moderate correlation.

This baseline model successfully demonstrates:
  Clear preprocessing, 
  Standard training workflow, 
  Proper evaluation metrics,

Correlation Heatmap

A correlation heatmap was generated to understand linear relationships:

```python
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=.5)
plt.title("Correlation Heatmap - California Housing Dataset")
plt.show()
```

Key Insights:

* `MedInc` has the strongest positive correlation with `MedHouseVal`
* `AveRooms`, `AveBedrms`, and `Population` have weaker correlations
* Latitude and longitude show spatial effects on pricing


Future Improvements

To make the project more advanced, we can add (which I will do in the next repo):

* Ridge and Lasso regression
* Hyperparameter tuning
* Polynomial features for non-linear relationships
* Residual analysis & predicted-vs-actual plots
* Build Flask or Streamlit app for predictions

Dependencies

```
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:

```bash
pip install -r requirements.txt
```

---

Contact: jawadhaider204@gmail.com
