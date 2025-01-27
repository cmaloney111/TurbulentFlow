

import random
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from xfoil import XFoil
from xfoil.model import Airfoil



airfoils_dir = "Stec8"
airfoils_dir_files = [airfoil_filename for airfoil_filename in os.listdir(airfoils_dir) if airfoil_filename.endswith('COR')]
random.shuffle(airfoils_dir_files)


for filename in airfoils_dir_files:
    coordinates = {'x': [], 'y': []}
    with open(os.path.join(airfoils_dir, filename), 'r') as f:
        next(f)
        print(filename)
        for line in f:
            x, y = line.split()
            coordinates["x"].append(x)
            coordinates["y"].append(y)

    xf = XFoil()
    airfoil = Airfoil(np.array(coordinates["x"]), np.array(coordinates["y"]))
    xf.airfoil = airfoil

    alphas_xfoil = np.linspace(-5, 15, 50)
    Re_values_to_test = [1e4, 8e4, 2e5, 1e6, 1e8]

    # Obtain data
    aeros = []

    for re in Re_values_to_test:
        xf.Re = re
        xf.max_iter = 1
        a, cl, cd, cm, cp = xf.aseq(-5, 15, 0.4)
        aeros.append({"CD": cd, "CL": cl})

    # print(aeros)
    for i in range(len(aeros)):
        xfoil_line2d, = plt.plot(
            aeros[i]["CD"],
            aeros[i]["CL"],
            linestyle=(0, (1, 1.5)), linewidth=2.2,
            # ".", markeredgewidth=0, markersize=4, alpha=0.8,
            zorder=5,
            label=Re_values_to_test[i]
        )
    plt.title(filename.split('.')[0])


