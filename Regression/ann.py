import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
joblib.dump(scaler, 'models/scaler_inputs_nn.pkl')

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2)  # Output layer for 2 target values
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define callbacks for early stopping and model checkpointing
checkpoint_path = 'models/best_neural_network_model.keras'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with validation
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# Load the best model after training
model.load_weights(checkpoint_path)

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

# Save the model architecture and weights
model.save('models/neural_network_model_final.keras')
