import pandas as pd
#Prep Dataset
housing_data = pd.read_csv("housing.csv")
housing_data.dropna(inplace=True)
housing_data.drop(['longitude', 'latitude'], axis=1, inplace=True)
housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'])
housing_data.rename(columns={'ocean_proximity_<1H OCEAN': '<1h ocean', 'ocean_proximity_INLAND': 'inland', 'ocean_proximity_ISLAND': 'island', 'ocean_proximity_NEAR BAY': 'near bay', 'ocean_proximity_NEAR OCEAN': 'near ocean'}, inplace=True)

#Train and test Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

X = housing_data.drop("median_house_value", axis=1)
y = housing_data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) 

########################################################################################################
#Train the Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_predictions = model.predict(X_test_scaled)
#Test the model 
from sklearn.metrics import mean_squared_error
import numpy as np

mse_LR = mean_squared_error(y_test, y_predictions)
rmse_LR = np.sqrt(mse_LR)

print("MSE:", mse_LR)
print("RMSE:", rmse_LR)
########################################################################################################
#Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=8, random_state=42)
tree_reg.fit(X_train_scaled, y_train)
y_train_pred = tree_reg.predict(X_train_scaled)
y_test_pred = tree_reg.predict(X_test_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")
########################################################################################################
#Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Best parameters from your grid search
params = {
    'n_estimators': 200,
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Create the model with best params
rf_best = RandomForestRegressor(**params)

# Train the model
rf_best.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = rf_best.predict(X_train)
y_test_pred = rf_best.predict(X_test)

# Calculate RMSE for train and test
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train RMSE (Random Forest with Best Params): {train_rmse}")
print(f"Test RMSE (Random Forest with Best Params): {test_rmse}")
########################################################################################################
print("============ RMSE Scores ============") 
print(f"Linear Regression: {rmse_LR}")
print(f"Decision Tree: {rmse_test}")
print(f"Random Forest: {test_rmse}")
print("=====================================")
# Scale the full dataset's features (excluding target) for models that require scaling
X_scaled_full = scaler.transform(X)

# Get predictions from all models
housing_data_with_preds = housing_data.copy()
housing_data_with_preds["LR Prediction"] = model.predict(X_scaled_full)
housing_data_with_preds["DT Prediction"] = tree_reg.predict(X_scaled_full)
housing_data_with_preds["RF Prediction"] = rf_best.predict(X)

cols = list(housing_data_with_preds.columns)

# Remove it from its current position and append it to the end
cols.remove('median_house_value')
cols.append('median_house_value')

# Reorder the DataFrame
housing_data_with_preds = housing_data_with_preds[cols]

print(housing_data_with_preds)
