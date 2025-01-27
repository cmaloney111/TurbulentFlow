import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Define constants for new input/output sizes
N_inputs = 140  # Input size with x/y coordinates and additional features
N_outputs = 2   # Output is lift and drag coefficients

cache_file = "models/neuralfoil-nn.pth"
n_hidden_layers = 4  
width = 128  
print("Cache file:", cache_file)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            torch.nn.Linear(N_inputs, width),
            torch.nn.SiLU(),
        ]
        for _ in range(n_hidden_layers):
            layers += [
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
            ]
        layers += [torch.nn.Linear(width, N_outputs)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load and preprocess the data
df = pd.read_csv('training_data_new.csv').dropna(subset=['x_0', 'y_0']).fillna(0)
df_inputs = df.drop(columns=['airfoil_name', 'lift_coefficient', 'drag_coefficient', 'Unnamed: 0'])
df_outputs = df[['lift_coefficient', 'drag_coefficient']]

df_train_inputs, df_test_inputs, df_train_outputs, df_test_outputs = train_test_split(
    df_inputs, df_outputs, test_size=0.1, random_state=42
)

# Scale the data
scaler_inputs = StandardScaler()
df_train_inputs_scaled = scaler_inputs.fit_transform(df_train_inputs)
df_test_inputs_scaled = scaler_inputs.transform(df_test_inputs)
scaler_outputs = StandardScaler()
df_train_outputs_scaled = scaler_outputs.fit_transform(df_train_outputs)
df_test_outputs_scaled = scaler_outputs.transform(df_test_outputs)

# Save scalers
joblib.dump(scaler_inputs, 'models/scaler_inputs_nf.pkl')
joblib.dump(scaler_outputs, 'models/scaler_outputs_nf.pkl')

# Convert scaled data to tensors
train_inputs = torch.tensor(df_train_inputs_scaled, dtype=torch.float32)
train_outputs = torch.tensor(df_train_outputs_scaled, dtype=torch.float32)
test_inputs = torch.tensor(df_test_inputs_scaled, dtype=torch.float32)
test_outputs = torch.tensor(df_test_outputs_scaled, dtype=torch.float32)

# Create DataLoaders with a small batch size
batch_size = 8
train_loader = DataLoader(TensorDataset(train_inputs, train_outputs), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_inputs, test_outputs), batch_size=batch_size)

net = Net().to(device)

# Define optimizer with no weight decay
learning_rate = 1e-5
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, verbose=True)

# Define loss function for lift and drag coefficients
def loss_function(y_pred, y_data):
    return torch.mean(torch.nn.functional.mse_loss(y_pred, y_data))

# Training loop with high epochs
print("Training...")
num_epochs = 20000  # Increased epochs
for epoch in range(num_epochs):
    net.train()
    for x, y_data in train_loader:
        x, y_data = x.to(device), y_data.to(device)
        loss = loss_function(net(x), y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation on the test set
    net.eval()
    test_loss_components = []
    with torch.no_grad():
        for x, y_data in test_loader:
            x, y_data = x.to(device), y_data.to(device)
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
