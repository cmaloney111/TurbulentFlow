import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv('training_data_stec8.csv')

# Clean data by removing rows with missing values in target columns
data = data.dropna(subset=['lift_coefficient', 'drag_coefficient'])

# Function to extract coordinates from the 'coordinates' column
def extract_coordinates(coord_string):
    coords = ast.literal_eval(coord_string)
    x_values = [x for x, y in coords]
    y_values = [y for x, y in coords]
    return x_values, y_values

# Apply the function to extract coordinates and add as new columns
data[['x_coords', 'y_coords']] = data['coordinates'].apply(lambda x: pd.Series(extract_coordinates(x)))

# Create DataFrames for x and y coordinates
x_coords_df = pd.DataFrame(data['x_coords'].tolist(), index=data.index)
y_coords_df = pd.DataFrame(data['y_coords'].tolist(), index=data.index)

# Rename the new columns
x_coords_df.columns = [f'x_{i}' for i in range(x_coords_df.shape[1])]
y_coords_df.columns = [f'y_{i}' for i in range(y_coords_df.shape[1])]

# Concatenate all the data into one DataFrame
data = pd.concat([data.reset_index(drop=True), x_coords_df, y_coords_df], axis=1)

# Drop unnecessary columns and prepare feature and target sets
data.drop(columns=['coordinates', 'x_coords', 'y_coords']).to_csv('training_data_new.csv')
X = data.drop(columns=['airfoil_name', 'coordinates', 'x_coords', 'y_coords', 'lift_coefficient', 'drag_coefficient'])
y = data[['lift_coefficient', 'drag_coefficient']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to reuse when predicting
joblib.dump(scaler, 'models/scaler_inputs_xgb.pkl')

# Initialize XGBoost regressor
model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# Define a more aggressive hyperparameter grid
param_dist = {
    'n_estimators': [1000, 2000, 3000],  # High number of estimators
    'max_depth': [10, 15, 20],  # Very deep trees for high complexity
    'learning_rate': [0.01, 0.05],  # Small learning rate, more boosting rounds
    'subsample': [1.0],  # No sampling
    'colsample_bytree': [1.0],  # No feature subsampling
    'gamma': [0.0],  # No regularization
    'min_child_weight': [1],  # Smallest value to allow splitting of trees
}

# # Define a broader hyperparameter grid for better search
# param_dist = {
#     'n_estimators': [200, 300, 400],  # More estimators
#     'max_depth': [3, 5, 7, 10],  # Deeper trees
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Wider range for learning rate
#     'subsample': [0.7, 0.8, 0.9, 1.0],  # More variety in subsample ratio
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # Sampling ratio for features
#     'gamma': [0, 0.1, 0.2, 0.3],  # Control overfitting by setting a minimum loss reduction
#     'min_child_weight': [1, 3, 5, 7],  # Minimum sum of instance weight needed in a child
# }


# Use RandomizedSearchCV to sample from the grid
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                   n_iter=50, cv=5, verbose=1, scoring='neg_mean_squared_error', 
                                   random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters found
print("Best parameters found: ", random_search.best_params_)

# Get the best model from the search
best_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the trained model
joblib.dump(best_model, 'models/first_xgboost_overfit.pkl')
