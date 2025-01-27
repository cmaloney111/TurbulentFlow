import random
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from xfoil import XFoil
from xfoil.model import Airfoil
from neuralfoil import get_aero_from_airfoil
import keras
import joblib
import matplotlib.cm as cm
import pandas as pd
import aerosandbox as asb
import matplotlib.pyplot as plt
import numpy as np
import ast
import torch

def plot_xfoil(csv_file, filename):
    df = pd.read_csv(csv_file)
    airfoil_data = df[df['airfoil_name'] == filename]
    
    reynolds_numbers = airfoil_data['reynolds_number'].unique()

    reynolds_to_angles = {
        int(reynolds): airfoil_data[airfoil_data['reynolds_number'] == reynolds]['angle_of_attack'].tolist()
        for reynolds in reynolds_numbers
    }

    cmap = cm.get_cmap('tab10', len(reynolds_numbers))  

    coordinates = {'x': [], 'y': []}
    with open(os.path.join("Stec8", filename + '.COR'), 'r') as f:
        next(f)
        for line in f:
            x, y = line.split()
            coordinates["x"].append(x)
            coordinates["y"].append(y)

    xf = XFoil()
    airfoil = Airfoil(np.array(coordinates["x"]), np.array(coordinates["y"]))
    xf.airfoil = airfoil

    # alphas_xfoil = np.linspace(-5, 15, 50)
    # Re_values_to_test = [1e4, 8e4, 2e5, 1e6, 1e8]

    # Obtain data
    aeros = []

    for re in reynolds_numbers:
        xf.Re = re
        xf.max_iter = 2000000
        cl_list, cd_list = [], []
        for alpha in reynolds_to_angles[re]:
            cl, cd, _, _ = xf.a(alpha) # cl, cd, cm, cp
            cl_list.append(cl)
            cd_list.append(cd)
        aeros.append({"CD": cd_list, "CL": cl_list})

    # print(aeros)
    for i in range(len(aeros)):
        plt.plot(
            aeros[i]["CD"],
            aeros[i]["CL"],
            linestyle='dotted', 
            linewidth=2.2,
            # ".", markeredgewidth=0, markersize=4, alpha=0.8,
            zorder=5,
            label="XFoil RE = " + str(round(reynolds_numbers[i], 0)),
            color = cmap(i)
        )
    # plt.title(filename.split('.')[0])




N_inputs = 140
N_outputs = 2
n_hidden_layers = 4
width = 128

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = Net().to(device)

learning_rate = 1e-4
optimizer = torch.optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=3e-5)

checkpoint = torch.load('models/neuralfoil-nn.pth')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()

def extract_coordinates(coord_string):
    coords = ast.literal_eval(coord_string)
    
    x_values = [x for x, y in coords]
    y_values = [y for x, y in coords]

    x_values = x_values + [0] * (69 - len(x_values))
    y_values = y_values + [0] * (69 - len(y_values))
    return x_values, y_values

def plot_airfoil_drag_lift(csv_file, airfoil_name, model_choice='rff'):
    df = pd.read_csv(csv_file)
    
    airfoil_data = df[df['airfoil_name'] == airfoil_name]
    if airfoil_data.empty:
        print(f"No data found for airfoil '{airfoil_name}'")
        return False
    
    reynolds_numbers = airfoil_data['reynolds_number'].unique()
    reynolds_to_angles = {
        int(reynolds): airfoil_data[airfoil_data['reynolds_number'] == reynolds]['angle_of_attack'].tolist()
        for reynolds in reynolds_numbers
    }
    shape_str = airfoil_data.iloc[0]['coordinates']
    airfoil_shape = np.array(ast.literal_eval(shape_str))  
    
    cmap = cm.get_cmap('tab10', len(reynolds_numbers))  

    
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

        plt.plot(x, y, linestyle='--', linewidth=2.2, alpha=0.3, label=f"GT Re = {reynolds:.0f}", color=color, zorder=5)
        plt.scatter(x, y, linestyle=(0, (1, 1.5)), color=color, alpha=0.3, linewidth=2.2, zorder=5)

        reynolds_data[['x_coords', 'y_coords']] = reynolds_data['coordinates'].apply(lambda x: pd.Series(extract_coordinates(x)))

        x_coords_df = pd.DataFrame(reynolds_data['x_coords'].tolist(), index=reynolds_data.index)
        y_coords_df = pd.DataFrame(reynolds_data['y_coords'].tolist(), index=reynolds_data.index)

        x_coords_df.columns = [f'x_{i}' for i in range(x_coords_df.shape[1])]
        y_coords_df.columns = [f'y_{i}' for i in range(y_coords_df.shape[1])]

        reynolds_data = pd.concat([reynolds_data.reset_index(drop=True), x_coords_df.reset_index(drop=True), y_coords_df.reset_index(drop=True)], axis=1)

        x = reynolds_data.drop(columns=['airfoil_name', 'coordinates', 'x_coords', 'y_coords', 'lift_coefficient', 'drag_coefficient'])

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
            preds = scaler_outputs.transform(list(zip(lift, drag)))
        elif model_choice == 'neuralfoil':
            x_numpy = x_coords_df.iloc[0].to_numpy()
            y_numpy = y_coords_df.iloc[0].to_numpy()
            coords_numpy = np.stack((x_numpy, y_numpy), axis=-1)
            coords_numpy = coords_numpy[~np.all(coords_numpy == 0, axis=1)]
            airfoil = asb.Airfoil(name=airfoil_name, coordinates=coords_numpy).to_kulfan_airfoil()
            preds = get_aero_from_airfoil(airfoil, np.array(reynolds_to_angles[reynolds]), reynolds, model_size='xxxlarge')
            preds = list(zip(preds['CL'], preds['CD']))
        else:
            print("Invalid model choice!")
            return

    
        lift = [row[0] for row in preds]
        drag = [row[1] for row in preds]

        plt.plot(drag, lift, linestyle=':', linewidth=2.2, label=f"Predicted Re = {reynolds:.0f}", color=color, zorder=10)
        plt.scatter(drag, lift, linewidth=2.2, color=color, zorder=10)
    

    plt.xscale('log')
    plt.legend()
    inset_ax = plt.axes([0.75, 0.8, 0.2, 0.2])  
    inset_ax.plot(airfoil_shape[:, 0], airfoil_shape[:, 1], color='black')
    inset_ax.set_title("Airfoil Shape")
    inset_ax.axis("equal")
    inset_ax.axis("off")
    return True
    

if __name__ == '__main__':
    airfoils_dir = "Stec8"
    airfoils_coords = [filename for filename in os.listdir(airfoils_dir) if filename.endswith('.COR')]
    random.shuffle(airfoils_coords)
    for method in ['neuralfoil', 'xgb']:
        for filename in airfoils_coords:
            print(filename)
            airfoil_name = filename.split('.')[0]
            csv_file = 'training_data_stec8.csv'
            plt.figure(figsize=(10, 8))
            plot_xfoil(csv_file, airfoil_name)
            print_plot = plot_airfoil_drag_lift(csv_file, airfoil_name, method)
            if print_plot:
                plt.savefig(f'figs/{method}/{airfoil_name}.png')




    # model_choice = input("Choose a model (rff, xgb, ann, net, neuralfoil): ").strip().lower()
    
    # # NACA0009
    # # CLARK-Y
    # plt.figure(figsize=(10, 8))
    # plot_xfoil('training_data_stec8.csv', 'NACA0009')
    # plot_airfoil_drag_lift('training_data_stec8.csv', 'NACA0009', model_choice)
    # plt.show()
    # # SD5060
    # # SD8000
    # # SD7037
    # # S2048