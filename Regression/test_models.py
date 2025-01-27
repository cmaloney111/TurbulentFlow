import keras
import joblib
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import torch

N_inputs = 140  # New input size with airfoil x/y coordinates and additional features
N_outputs = 2   # Output is now just lift and drag coefficients
n_hidden_layers = 4
width = 128

# Load the models
ann = keras.saving.load_model("models/best_neural_network_model.keras")

with open('models/first_random_forest.pkl', 'rb') as f:
    rff = joblib.load(f)

with open('models/first_xgboost.pkl', 'rb') as f:
    xgb = joblib.load(f)

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


# Assuming pre-loaded training/testing data and scaled means/covariances:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# net = Net(
#     mean_inputs_scaled=torch.tensor(mean_inputs_scaled, dtype=torch.float32).to(device),
#     cov_inputs_scaled=torch.tensor(cov_inputs_scaled, dtype=torch.float32).to(device),
# ).to(device)
net = Net().to(device)

# Define the optimizer
learning_rate = 1e-4
optimizer = torch.optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=3e-5)

checkpoint = torch.load('models/neuralfoil-nn.pth')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Put the model in evaluation mode
net.eval()

# Load the scaler from the file


# Define a function to extract coordinates from string
def extract_coordinates(coord_string):
    coords = ast.literal_eval(coord_string)
    
    x_values = [x for x, y in coords]
    y_values = [y for x, y in coords]

    x_values = x_values + [0] * (69 - len(x_values))
    y_values = y_values + [0] * (69 - len(y_values))
    return x_values, y_values

# Define the main function to plot airfoil data and make predictions
def plot_airfoil_drag_lift(csv_file, airfoil_name, model_choice='rff'):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Filter data by airfoil name
    airfoil_data = df[df['airfoil_name'] == airfoil_name]
    if airfoil_data.empty:
        print(f"No data found for airfoil '{airfoil_name}'")
        return
    
    reynolds_numbers = airfoil_data['reynolds_number'].unique()
    shape_str = airfoil_data.iloc[0]['coordinates']
    airfoil_shape = np.array(ast.literal_eval(shape_str))  
    
    cmap = cm.get_cmap('tab10', len(reynolds_numbers))  

    plt.figure(figsize=(10, 8))
    plt.title(f"Lift vs. Drag Coefficient for {airfoil_name}")
    plt.xlabel("Drag Coefficient ")
    plt.ylabel("Lift Coefficient")

    for i, reynolds in enumerate(reynolds_numbers):
        reynolds_data = airfoil_data[airfoil_data['reynolds_number'] == reynolds]
        
        reynolds_data = reynolds_data.drop_duplicates(subset='drag_coefficient')
        reynolds_data = reynolds_data.sort_values(by='lift_coefficient')
     
        x = reynolds_data['drag_coefficient']
        y = reynolds_data['lift_coefficient']
       
        color = cmap(i)

        # Plot ground truth (GT) data
        plt.plot(x, y, linestyle='--', linewidth=2.2, alpha=0.3, label=f"GT Re = {reynolds:.0f}", color=color, zorder=5)
        plt.scatter(x, y, linestyle=(0, (1, 1.5)), color=color, alpha=0.3, linewidth=2.2, zorder=5)

        # Extract coordinates from airfoil data
        reynolds_data[['x_coords', 'y_coords']] = reynolds_data['coordinates'].apply(lambda x: pd.Series(extract_coordinates(x)))

        x_coords_df = pd.DataFrame(reynolds_data['x_coords'].tolist(), index=reynolds_data.index)
        y_coords_df = pd.DataFrame(reynolds_data['y_coords'].tolist(), index=reynolds_data.index)

        x_coords_df.columns = [f'x_{i}' for i in range(x_coords_df.shape[1])]
        y_coords_df.columns = [f'y_{i}' for i in range(y_coords_df.shape[1])]

        reynolds_data = pd.concat([reynolds_data.reset_index(drop=True), x_coords_df.reset_index(drop=True), y_coords_df.reset_index(drop=True)], axis=1)

        # Drop columns not needed for prediction
        x = reynolds_data.drop(columns=['airfoil_name', 'coordinates', 'x_coords', 'y_coords', 'lift_coefficient', 'drag_coefficient'])

        # Select the model based on user input
        # print(x)
        if model_choice == 'rff':
            scaler_inputs = joblib.load('models/scaler_inputs_rf.pkl')
            x = scaler_inputs.transform(x)          
            preds = rff.predict(x)
        elif model_choice == 'xgb':
            scaler_inputs = joblib.load('models/scaler_inputs_xgb.pkl')
            x = scaler_inputs.transform(x)
            preds = xgb.predict(x)
        elif model_choice == 'ann':
            scaler_inputs = joblib.load('models/scaler_inputs_nn.pkl')
            x = scaler_inputs.transform(x)
            preds = ann.predict(x)
        elif model_choice == 'net':

            scaler_inputs = joblib.load('models/scaler_inputs_nf.pkl')
            scaler_outputs = joblib.load('models/scaler_outputs_nf.pkl')
            x = scaler_inputs.transform(x)
            preds = net(torch.tensor(x, dtype=torch.float32))
            lift = [torch.detach(row[0]) for row in preds]
            drag = [torch.detach(row[1]) for row in preds]
            lift_drag = scaler_outputs.transform(list(zip(lift, drag)))
            lift = [lift[0] for lift in lift_drag]
            drag = [drag[1] for drag in lift_drag]
        else:
            print("Invalid model choice!")
            return

        if model_choice != 'net':
            lift = [row[0] for row in preds]
            drag = [row[1] for row in preds]
        print(lift)
        print(drag)
        # Plot predictions
        plt.plot(drag, lift, linestyle=':', linewidth=2.2, label=f"Predicted Re = {reynolds:.0f}", color=color, zorder=10)
        plt.scatter(drag, lift, linewidth=2.2, color=color, zorder=10)

    plt.legend()
    
    # Inset for airfoil shape
    inset_ax = plt.axes([0.75, 0.8, 0.2, 0.2])  
    inset_ax.plot(airfoil_shape[:, 0], airfoil_shape[:, 1], color='black')
    inset_ax.set_title("Airfoil Shape")
    inset_ax.axis("equal")
    inset_ax.axis("off")
    
    # Show the plot
    plt.show()

# Main block to run the function
if __name__ == '__main__':
    # Ask the user to choose a model
    model_choice = input("Choose a model (rff, xgb, ann, net): ").strip().lower()
    
    # Call the function to plot and predict
    # NACA0009
    # CLARK-Y
    plot_airfoil_drag_lift('training_data_stec8.csv', 'NACA0009', model_choice)
    # SD5060
    # SD8000
    # SD7037
    # S2048
