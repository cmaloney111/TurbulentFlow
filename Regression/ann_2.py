import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import ast

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
joblib.dump(scaler, 'models/scaler_inputs_ann.pkl')

# Define the neural network architecture to encourage overfitting
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Add more layers with very high neuron counts
    x = Dense(2048, activation='relu')(inputs)
    x = Dense(2048, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    # Output layer
    outputs = Dense(y_train.shape[1], activation='linear')(x)

    model = Model(inputs, outputs)
    return model

# Initialize and compile the model
input_shape = X_train_scaled.shape[1]
model = build_model(input_shape=(input_shape,))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')  # Higher learning rate

# Checkpoint to save the model
checkpoint = ModelCheckpoint('models/ann_overfit.keras', save_best_only=True, monitor='loss', mode='min')

# Train the model on the training set with low batch size and without validation split
history = model.fit(
    X_train_scaled, y_train,
    epochs=500,                # Lower epochs but a high learning rate to quickly reach low loss
    batch_size=1,              # Batch size of 1 to focus on each sample individually
    verbose=1,
    callbacks=[checkpoint]
)

# Evaluate on the test set to check for overfitting effect
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the final model and scaler
model.save('models/ann_final.h5')
