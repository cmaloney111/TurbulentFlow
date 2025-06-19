import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Define constants matching reference NeuralFoil architecture
N_inputs = 140  # Input size with x/y coordinates and additional features
N_outputs = 2   # Output is lift and drag coefficients

cache_file = "models/neuralfoil-nn.pth"
n_hidden_layers = 5  # Updated to match reference (5 hidden layers)
width = 512  # Updated to match reference (512 neurons per layer)
print("Cache file:", cache_file)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Build network layers with batch normalization and dropout for stability
        layers = [
            torch.nn.Linear(N_inputs, width),
            torch.nn.BatchNorm1d(width),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1)
        ]
        for _ in range(n_hidden_layers):
            layers += [
                torch.nn.Linear(width, width),
                torch.nn.BatchNorm1d(width),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.1)
            ]
        layers += [torch.nn.Linear(width, N_outputs)]
        self.net = torch.nn.Sequential(*layers)
        
        # Initialize weights similar to reference implementation
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """Forward pass with optional symmetric evaluation"""
        return self.net(x)
    
    def forward_symmetric(self, x: torch.Tensor):
        """
        Symmetric forward pass inspired by NeuralFoil reference implementation.
        Evaluates both normal and flipped inputs, then averages results.
        """
        # Normal forward pass
        normal_output = self.forward(x)
        
        # For airfoil data, we can flip coordinates to create symmetric evaluation
        # This is a simplified version - in practice you'd need to properly flip
        # the airfoil coordinates while keeping Reynolds and angle of attack
        x_flipped = x.clone()
        # Flip y-coordinates (assuming they are in positions 69-137)
        if x.shape[1] >= 138:  # 69 x-coords + 69 y-coords + 2 other features
            x_flipped[:, 69:138] = -x_flipped[:, 69:138]  # Flip y-coordinates
        
        flipped_output = self.forward(x_flipped)
        
        # Average the outputs (with some adjustments for lift coefficient sign)
        averaged_output = normal_output.clone()
        averaged_output[:, 0] = (normal_output[:, 0] - flipped_output[:, 0]) / 2  # CL should be antisymmetric
        averaged_output[:, 1] = (normal_output[:, 1] + flipped_output[:, 1]) / 2  # CD should be symmetric
        
        return averaged_output

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

# Define optimizer matching reference implementation (RAdam with weight decay)
learning_rate = 1e-4
optimizer = torch.optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=3e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)

# Define sophisticated loss function inspired by reference implementation
def loss_function(y_pred, y_data):
    """
    Sophisticated loss function with weighted components for lift and drag coefficients.
    Uses Huber loss for robustness to outliers.
    """
    # Separate lift and drag coefficients
    cl_pred, cd_pred = y_pred[:, 0], y_pred[:, 1]
    cl_true, cd_true = y_data[:, 0], y_data[:, 1]
    
    # Use Huber loss for robustness (similar to reference implementation)
    cl_loss = torch.nn.functional.huber_loss(cl_pred, cl_true, delta=0.1)
    cd_loss = torch.nn.functional.huber_loss(cd_pred, cd_true, delta=0.01)
    
    # Weight the losses (drag coefficient typically has smaller magnitude)
    total_loss = cl_loss + 2.0 * cd_loss  # Give drag loss higher weight
    
    return total_loss

# Training loop with improvements inspired by reference implementation
print("Training...")
num_epochs = 1000  # Increased epochs to match reference
best_test_loss = float('inf')
patience_counter = 0
patience_limit = 100

for epoch in range(num_epochs):
    # Training phase
    net.train()
    train_losses = []
    
    for x, y_data in train_loader:
        x, y_data = x.to(device), y_data.to(device)
        
        # Use symmetric forward pass occasionally for better training
        if epoch > 100 and torch.rand(1) < 0.3:  # 30% chance after epoch 100
            y_pred = net.forward_symmetric(x)
        else:
            y_pred = net(x)
            
        loss = loss_function(y_pred, y_data)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_losses.append(loss.item())

    # Evaluation on the test set
    net.eval()
    test_losses = []
    with torch.no_grad():
        for x, y_data in test_loader:
            x, y_data = x.to(device), y_data.to(device)
            
            # Use symmetric evaluation for testing
            y_pred = net.forward_symmetric(x)
            test_loss = loss_function(y_pred, y_data)
            test_losses.append(test_loss.item())
    
    avg_train_loss = np.mean(train_losses)
    avg_test_loss = np.mean(test_losses)
    
    print(f"Epoch: {epoch:4d} | Train Loss: {avg_train_loss:.6g} | Test Loss: {avg_test_loss:.6g}")
    
    # Learning rate scheduling
    scheduler.step(avg_test_loss)
    
    # Early stopping with patience
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        patience_counter = 0
        # Save best model checkpoint
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'best_test_loss': best_test_loss
        }, cache_file)
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience_limit:
        print(f"Early stopping at epoch {epoch} - no improvement for {patience_limit} epochs")
        break

print(f"Training completed. Best test loss: {best_test_loss:.6g}")
