import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
data = pd.read_csv('training_data_stec8.csv')

# Clean data by removing rows with missing values in target columns
data = data.dropna(subset=['lift_coefficient', 'drag_coefficient', 'coordinates'])

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
X = data.drop(columns=['airfoil_name', 'coordinates', 'x_coords', 'y_coords', 'lift_coefficient', 'drag_coefficient'])
y = data[['lift_coefficient', 'drag_coefficient']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to reuse when predicting
joblib.dump(scaler, 'models/scaler_inputs_rf.pkl')

model = RandomForestRegressor(
    n_estimators=5000,          # Very large number of trees
    max_depth=None,             # Allow trees to grow as deep as possible
    min_samples_split=2,        # No minimum for splitting
    min_samples_leaf=1,         # No minimum number of samples required in leaf
    max_features=None,          # No restriction on number of features
    random_state=42,            # For reproducibility
    n_jobs=-1                   # Use all cores for faster training
)

# model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the scaled data
model.fit(X_train_scaled, y_train)

# Perform cross-validation to assess model's complexity
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the trained model
joblib.dump(model, 'models/first_random_forest_overfit.pkl')
