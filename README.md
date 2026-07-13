# python-project-5-datacleaning-and-supervised-machine-learning


## Project Description
This project performs comprehensive data preprocessing and trains regression models on CSV data using Python.

## Dataset
- **File:** 'match_predictions.csv'
- **Type:** Tabular data loaded using pandas

## Steps Performed

### 1. Data Loading & Initial Exploration
- Imported pandas, numpy, matplotlib libraries
- Loaded CSV dataset using `pd.read_csv()`
- Displayed data overview with `.head()`

### 2. Data Cleaning
- **Dropped unnecessary data:** Removed irrelevant rows/columns
- **Converted text to datetime:** Used `pd.to_datetime()` for date columns
- **One-Hot Encoding:** Applied `pd.get_dummies()` to convert categorical variables to numerical,checked for missing values

### 3. Feature & Target Separation
- **X (Features):** Created feature matrix with independent variables
- **y (Target/Labels):** Created target variable for prediction

### 4. Train-Test Split
- Used `train_test_split()` to divide data into training and testing sets
- Typically 70% training, 30% testing

### 5. Feature Scaling
- Applied **StandardScaler** to normalize features
- Fitted scaler on training data, transformed both train and test sets

### 6. Model Training
- **Random Forest Regressor:** Ensemble learning method using multiple decision trees
- **Gradient Boosting Regressor:** Sequential ensemble method that builds trees to correct errors


## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
