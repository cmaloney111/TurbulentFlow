import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools.code_benchmarking import time_function
from neuralfoil import get_aero_from_airfoil

airfoil = asb.Airfoil("dae11")
airfoil = airfoil.repanel().normalize()
Re = 1e6

for model_size in ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]:

    n_runs = 100000


    def func():
        alpha = np.linspace(0, 10, n_runs)
        get_aero_from_airfoil(
            airfoil=airfoil,
            alpha=alpha,
            Re=Re,
            model_size=model_size
        )


    time_total, _ = time_function(func, desired_runtime=1)
    print(model_size, time_total)

# def func():
#     alpha = np.linspace(0, 10, 200)
#     np.random.shuffle(alpha)
#     xf = asb.XFoil(
#         airfoil=airfoil,
#         Re=Re
#     ).alpha(alpha)
#
# time_total, _ = time_function(func)
# print("XFoil", time_total)
