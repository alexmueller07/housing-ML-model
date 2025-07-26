import pandas as pd

# ===============================
# Load and preprocess the dataset
# ===============================

# Load housing dataset from CSV
housing_data = pd.read_csv("housing.csv")

# Drop rows with missing values
housing_data.dropna(inplace=True)

# Drop location-related columns (they are not used in this analysis)
housing_data.drop(['longitude', 'latitude'], axis=1, inplace=True)

# One-hot encode the 'ocean_proximity' categorical feature
housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'])

# Rename encoded columns to simpler names
housing_data.rename(columns={
    'ocean_proximity_<1H OCEAN': '<1h ocean',
    'ocean_proximity_INLAND': 'inland',
    'ocean_proximity_ISLAND': 'island',
    'ocean_proximity_NEAR BAY': 'near bay',
    'ocean_proximity_NEAR OCEAN': 'near ocean'
}, inplace=True)

# ===============================
# Train-test split and scaling
# ===============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

# Separate features (X) and target variable (y)
X = housing_data.drop("median_house_value", axis=1)
y = housing_data["median_house_value"]

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize feature values (important for models like linear regression)
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Linear Regression Model
# ===============================

from sklearn.linear_model import LinearRegression

# Create and train linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict housing prices on test set
y_predictions = model.predict(X_test_scaled)

# Evaluate model performance
from sklearn.metrics import mean_squared_error
import numpy as np

mse_LR = mean_squared_error(y_test, y_predictions)
rmse_LR = np.sqrt(mse_LR)

print("MSE:", mse_LR)
print("RMSE:", rmse_LR)

# ===============================
# Decision Tree Regressor
# ===============================

from sklearn.tree import DecisionTreeRegressor

# Create decision tree model with limited depth to prevent overfitting
tree_reg = DecisionTreeRegressor(max_depth=8, random_state=42)
tree_reg.fit(X_train_scaled, y_train)

# Predict on train and test data
y_train_pred = tree_reg.predict(X_train_scaled)
y_test_pred = tree_reg.predict(X_test_scaled)

# Calculate RMSE for training and test data
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")

# ===============================
# Random Forest Regressor
# ===============================

from sklearn.ensemble import RandomForestRegressor

# Use a predefined set of hyperparameters (e.g. from grid search)
params = {
    'n_estimators': 200,
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Create and train the random forest model
rf_best = RandomForestRegressor(**params)
rf_best.fit(X_train, y_train)  # Note: unscaled input used for random forest

# Predict on both training and test data
y_train_pred = rf_best.predict(X_train)
y_test_pred = rf_best.predict(X_test)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train RMSE (Random Forest with Best Params): {train_rmse}")
print(f"Test RMSE (Random Forest with Best Params): {test_rmse}")

# ===============================
# Final Summary
# ===============================

print("============ RMSE Scores ============") 
print(f"Linear Regression: {rmse_LR}")
print(f"Decision Tree: {rmse_test}")
print(f"Random Forest: {test_rmse}")
print("=====================================")

# ===============================
# Model Predictions on Full Dataset
# ===============================

# Apply standardization to full dataset (excluding target)
X_scaled_full = scaler.transform(X)

# Create a copy of the full dataset to hold predictions
housing_data_with_preds = housing_data.copy()

# Add predictions from all models
housing_data_with_preds["LR Prediction"] = model.predict(X_scaled_full)
housing_data_with_preds["DT Prediction"] = tree_reg.predict(X_scaled_full)
housing_data_with_preds["RF Prediction"] = rf_best.predict(X)

# Move the actual house value column to the end of the DataFrame
cols = list(housing_data_with_preds.columns)
cols.remove('median_house_value')
cols.append('median_house_value')
housing_data_with_preds = housing_data_with_preds[cols]

# Display dataset with predictions
print(housing_data_with_preds)
