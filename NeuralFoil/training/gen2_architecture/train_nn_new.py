import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define constants for new input/output sizes
N_inputs = 140  # New input size with airfoil x/y coordinates and additional features
N_outputs = 2   # Output is now just lift and drag coefficients

cache_file = "nn-test-updated.pth"
n_hidden_layers = 3
width = 64
print("Cache file:", cache_file)

# Define the new model
class Net(torch.nn.Module):
    def __init__(self, mean_inputs_scaled=None, cov_inputs_scaled=None):
        super().__init__()

        # Scaling variables
        # self.mean_inputs_scaled = mean_inputs_scaled
        # self.cov_inputs_scaled = cov_inputs_scaled
        # self.inv_cov_inputs_scaled = torch.inverse(cov_inputs_scaled)

        # Define the network layers
        layers = [
            torch.nn.Linear(N_inputs, width),
            torch.nn.SiLU(),
        ]
        for i in range(n_hidden_layers):
            layers += [
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
            ]
        layers += [torch.nn.Linear(width, N_outputs)]
        self.net = torch.nn.Sequential(*layers)

    # def squared_mahalanobis_distance(self, x: torch.Tensor):
    #     return torch.sum(
    #         (x - self.mean_inputs_scaled) @ self.inv_cov_inputs_scaled * (x - self.mean_inputs_scaled),
    #         dim=1
    #     )

    def forward(self, x: torch.Tensor):
        # Normal forward pass
        y = self.net(x)
        return y


# Assuming pre-loaded training/testing data and scaled means/covariances:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Prepare DataLoader objects for training and testing

# Load the data
df = pd.read_csv('../../../Regression/training_data_new.csv')
df = df.dropna(subset=['x_0', 'y_0'])
df = df.fillna(0)


# Define inputs and outputs
df_inputs = df.drop(columns=['airfoil_name', 'lift_coefficient', 'drag_coefficient', 'Unnamed: 0'])
df_outputs = df[['lift_coefficient', 'drag_coefficient']]

# Train-test split
df_train_inputs, df_test_inputs, df_train_outputs, df_test_outputs = train_test_split(
    df_inputs, df_outputs, test_size=0.1, random_state=42
)

# Scaling the inputs
scaler_inputs = StandardScaler()
df_train_inputs_scaled = scaler_inputs.fit_transform(df_train_inputs)
df_test_inputs_scaled = scaler_inputs.transform(df_test_inputs)

# Scaling the outputs
scaler_outputs = StandardScaler()
df_train_outputs_scaled = scaler_outputs.fit_transform(df_train_outputs)
df_test_outputs_scaled = scaler_outputs.transform(df_test_outputs)

import joblib

joblib.dump(scaler_inputs, 'scaler_inputs.pkl')
joblib.dump(scaler_outputs, 'scaler_outputs.pkl')

# Convert scaled data to tensors
train_inputs = torch.tensor(df_train_inputs_scaled, dtype=torch.float32)
train_outputs = torch.tensor(df_train_outputs_scaled, dtype=torch.float32)
test_inputs = torch.tensor(df_test_inputs_scaled, dtype=torch.float32)
test_outputs = torch.tensor(df_test_outputs_scaled, dtype=torch.float32)

# mean_inputs_scaled = np.mean(df_train_inputs_scaled, axis=0)
# print(df_train_inputs_scaled)
# cov_inputs_scaled = np.cov(df_train_inputs_scaled, rowvar=False)
# # print(cov_inputs_scaled)

# Create DataLoaders
batch_size = 256
train_loader = DataLoader(
    dataset=TensorDataset(train_inputs, train_outputs),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=TensorDataset(test_inputs, test_outputs),
    batch_size=batch_size
)

# net = Net(
#     mean_inputs_scaled=torch.tensor(mean_inputs_scaled, dtype=torch.float32).to(device),
#     cov_inputs_scaled=torch.tensor(cov_inputs_scaled, dtype=torch.float32).to(device),
# ).to(device)
net = Net().to(device)

# Define the optimizer
learning_rate = 1e-4
optimizer = torch.optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=3e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, verbose=True)

# Try loading an existing model
# try:
#     checkpoint = torch.load(cache_file)
#     net.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     print("Model found, resuming training.")
# except FileNotFoundError:
#     print("No existing model found, starting fresh.")


# Optional: Print shapes to verify
print(f"Train Inputs Shape: {train_inputs.shape}")
print(f"Train Outputs Shape: {train_outputs.shape}")
print(f"Test Inputs Shape: {test_inputs.shape}")
print(f"Test Outputs Shape: {test_outputs.shape}")

# Define loss function for lift and drag coefficients
def loss_function(y_pred, y_data):
    return torch.mean(torch.nn.functional.huber_loss(y_pred, y_data, delta=0.05))

# Training loop
print("Training...")
num_epochs = 100000000
for epoch in range(num_epochs):
    net.train()
    for x, y_data in train_loader:
        x = x.to(device)
        y_data = y_data.to(device)

        # Compute loss and perform backpropagation
        loss = loss_function(y_pred=net(x), y_data=y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation on the test set
    net.eval()
    test_loss_components = []
    for i, (x, y_data) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            y_data = y_data.to(device)
            y_pred = net(x)
            test_loss_components.append(loss_function(y_pred, y_data))

    test_loss = torch.mean(torch.stack(test_loss_components))
    print(f"Epoch: {epoch} | Train Loss: {loss.item():.6g} | Test Loss: {test_loss.item():.6g}")
    scheduler.step(test_loss)

    # Save model checkpoint
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, cache_file)
