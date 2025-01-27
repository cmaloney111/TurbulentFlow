import os
import math
import random
import subprocess
import shutil
import argparse
import numpy as np
from typing import List


# Constants and configurations
SAMPLES: int = 100
FREESTREAM_ANGLE: float = math.pi / 8
FREESTREAM_LENGTH: float = 10.0
FREESTREAM_LENGTH_FACTOR: float = 10.0
AIRFOIL_DATABASE: str = "./airfoil_database/"
OUTPUT_DIR: str = "./output/"
RESOLUTION: int = 128

# Random seed setup
seed: int = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print(f"Seed: {seed}")


def gen_mesh(airfoil_file: str, dim: int = 2) -> int:
    """Generate mesh for the given airfoil file."""
    airfoil_name = os.path.splitext(os.path.basename(airfoil_file))[0]
    try:
        airfoil_data = np.loadtxt(airfoil_file, skiprows=1)
    except Exception as e:
        print(f"Error reading airfoil file: {e}")
        return -1

    # Remove duplicate endpoint (if duplicate endpoint exists)
    if np.max(np.abs(airfoil_data[0] - airfoil_data[-1])) < 1e-6:
        airfoil_data = airfoil_data[:-1]

    points_output = ""
    point_index = 1000
    for x, y in airfoil_data:
        points_output += f"Point({point_index}) = {{ {x}, {y}, 0.0, 0.005 }};\n"
        point_index += 1

    # Generate the geo file from template
    try:
        with open("airfoil_template.geo", "rt") as in_file, open("airfoil.geo", "wt") as out_file:
            for line in in_file:
                line = line.replace("POINTS", points_output)
                line = line.replace("LAST_POINT_INDEX", str(point_index - 1))
                out_file.write(line)
    except FileNotFoundError as e:
        print(f"Error accessing template files: {e}")
        return -1

    # Run mesh generation using GMSH
    if subprocess.run(f"gmsh airfoil.geo -{dim} -format msh2 -order 2 -o airfoil.msh > /dev/null", shell=True, check=True).returncode != 0:
        print("Error during mesh creation!")
        return -1

    # Run conversion using gmsh2nek
    process = subprocess.Popen(
        ["gmsh2nek"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    responses = [str(dim), "airfoil", "0", "0", airfoil_name]
    stdout, stderr = process.communicate(input="\n".join(responses) + "\n")

    # Handle the output file
    try:
        shutil.move(f"{airfoil_name}.re2", OUTPUT_DIR)
    except shutil.Error:
        shutil.rmtree(f"{airfoil_name}.re2", ignore_errors=True)

    return 0


def run_sim(freestream_x: float, freestream_y: float) -> None:
    """Run the simulation using the given freestream velocity components."""
    try:
        with open("U_template", "rt") as in_file, open("0/U", "wt") as out_file:
            for line in in_file:
                line = line.replace("VEL_X", f"{freestream_x}")
                line = line.replace("VEL_Y", f"{freestream_y}")
                out_file.write(line)
    except FileNotFoundError as e:
        print(f"Error opening template files: {e}")
        return

    # Run the OpenFOAM solver
    os.system("./Allclean && simpleFoam > foam.log")


def output_processing(
    basename: str,
    freestream_x: float,
    freestream_y: float,
    data_dir: str = OUTPUT_DIR,
    pfile: str = 'OpenFOAM/postProcessing/internalCloud/500/cloud_p.xy',
    ufile: str = 'OpenFOAM/postProcessing/internalCloud/500/cloud_U.xy',
    res: int = RESOLUTION,
    image_index: int = 0
) -> None:
    """Process simulation output and save results."""
    np_output = np.zeros((6, res, res))

    # Read pressure file
    try:
        pressure_data = np.loadtxt(pfile)
    except Exception as e:
        print(f"Error reading pressure file: {e}")
        return

    # Read velocity file
    try:
        velocity_data = np.loadtxt(ufile)
    except Exception as e:
        print(f"Error reading velocity file: {e}")
        return

    # Fill output arrays with data
    cur_index = 0
    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(pressure_data[cur_index][0] - xf) < 1e-4 and abs(pressure_data[cur_index][1] - yf) < 1e-4:
                np_output[3][x][y] = pressure_data[cur_index][3]
                np_output[0][x][y] = freestream_x
                np_output[1][x][y] = freestream_y
                cur_index += 1
            else:
                np_output[3][x][y] = 0
                np_output[2][x][y] = 1.0

    # Fill velocity fields
    cur_index = 0
    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(velocity_data[cur_index][0] - xf) < 1e-4 and abs(velocity_data[cur_index][1] - yf) < 1e-4:
                np_output[4][x][y] = velocity_data[cur_index][3]
                np_output[5][x][y] = velocity_data[cur_index][4]
                cur_index += 1
            else:
                np_output[4][x][y] = 0
                np_output[5][x][y] = 0

    # Save the output data
    file_name = os.path.join(data_dir, f"{basename}_{int(freestream_x * 100)}_{int(freestream_y * 100)}.npz")
    print(f"Saving data to {file_name}")
    np.savez_compressed(file_name, a=np_output)


def main(args: argparse.Namespace) -> None:
    """Main function to run the simulation for a number of samples."""
    files = sorted(os.listdir(AIRFOIL_DATABASE))
    if not files:
        print(f"Error: No airfoils found in {AIRFOIL_DATABASE}")
        return

    # Main simulation loop
    for n in range(SAMPLES):
        print(f"Run {n}:")
        file_number = np.random.randint(0, len(files))
        basename = os.path.splitext(os.path.basename(files[file_number]))[0]
        print(f"\tUsing {files[file_number]}")

        # Freestream conditions
        length = FREESTREAM_LENGTH * np.random.uniform(1.0, FREESTREAM_LENGTH_FACTOR)
        angle = np.random.uniform(-FREESTREAM_ANGLE, FREESTREAM_ANGLE)
        fs_x = math.cos(angle) * length
        fs_y = -math.sin(angle) * length

        print(f"\tUsing length {length:.3f}, angle {angle:+.3f}")
        print(f"\tResulting freestream velocity: {fs_x}, {fs_y}")

        # Generate mesh for the current airfoil
        if gen_mesh(os.path.join(AIRFOIL_DATABASE, files[file_number]), args.dim) != 0:
            print("\tMesh generation failed, skipping")
            continue

        # Run the simulation
        # run_sim(fs_x, fs_y)

        # Process the output
        # output_processing(basename, fs_x, fs_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Airfoil re2 generator")
    parser.add_argument('-d', '--dim', choices=[2, 3], type=int, required=True, help="Mesh dimension (2 or 3)")
    args = parser.parse_args()

    main(args)
