import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import ast

def plot_airfoil_drag_lift(csv_file, airfoil_name):
    
    df = pd.read_csv(csv_file)
    
    
    airfoil_data = df[df['airfoil_name'] == airfoil_name]
    if airfoil_data.empty:
        print(f"No data found for airfoil '{airfoil_name}'")
        return
    
    
    reynolds_numbers = airfoil_data['reynolds_number'].unique()
    shape_str = airfoil_data.iloc[0]['coordinates']
    airfoil_shape = np.array(ast.literal_eval(shape_str))  
    
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Lift vs. Drag Coefficient for {airfoil_name}")
    plt.xlabel("Drag Coefficient ")
    plt.ylabel("Lift Coefficient")
    
    
    for reynolds in reynolds_numbers:
        reynolds_data = airfoil_data[airfoil_data['reynolds_number'] == reynolds]
        
        
        reynolds_data = reynolds_data.drop_duplicates(subset='drag_coefficient')
        reynolds_data = reynolds_data.sort_values(by='lift_coefficient')
     
        y = reynolds_data['lift_coefficient']
        x = reynolds_data['drag_coefficient']

        
        plt.plot(x, y, linestyle='--', linewidth=2.2, label=f"Re = {reynolds:.0f}", zorder=5)
        plt.scatter(
            reynolds_data['drag_coefficient'],
            reynolds_data['lift_coefficient'],
            linestyle=(0, (1, 1.5)), linewidth=2.2,
            zorder=5,
        )
    plt.legend()
    
    
    inset_ax = plt.axes([0.75, 0.8, 0.2, 0.2])  
    inset_ax.plot(airfoil_shape[:, 0], airfoil_shape[:, 1], color='black')
    inset_ax.set_title("Airfoil Shape")
    inset_ax.axis("equal")
    inset_ax.axis("off")
    
    plt.show()


if __name__ == '__main__':
    plot_airfoil_drag_lift('training_data_stec8.csv', 'DAE51')
