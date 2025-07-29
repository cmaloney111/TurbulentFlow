import os
import argparse
import numpy as np
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt

def visualize_coords(coords, title="Airfoil Shape", show_points=True):
    """
    Visualizes a list of 2D coordinates.

    Parameters:
    - coords: List or numpy array of (x, y) pairs
    - title: Title for the plot
    - show_points: If True, shows scatter points in addition to the line
    """
    coords = np.array(coords)
    x, y = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, '-', label='Shape')
    if show_points:
        plt.scatter(x, y, color='red', s=10, label='Points')
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def read_airfoil(path):
    with open(path, 'r') as f:
        dat_lines = f.readlines()
    header = dat_lines[0]
    lines = dat_lines[1:]
    coords = []
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 2:
                x, y = map(float, parts)
                coords.append((x, y))
    return header, np.array(coords)

def remove_duplicate_points(coords):
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return coords[np.sort(unique_indices)]

def find_leading_edge(coords):
    return np.argmin(coords[:, 0])  # min x usually marks LE

def split_surfaces(coords):
    le_idx = find_leading_edge(coords)
    upper = coords[:le_idx+1]
    lower = coords[le_idx:][::-1]  # reverse to ensure TE to LE
    
    return upper, lower

def enforce_trailing_edge(upper, lower):
    # Original first points (at the TE) from both surfaces
    upper_te = upper[0]
    lower_te = lower[0]

    # Average y and fix x to 1.0
    avg_y = (upper_te[1] + lower_te[1]) / 2
    te_point = np.array([1.0, avg_y])

    # Replace the first points of both with this new TE point
    upper[0] = te_point
    lower[0] = te_point

    return upper, lower


def merge_and_resample(upper, lower, n_points=200):
    coords = np.vstack((upper, lower[::-1]))  # avoid duplicate TE
    coords = remove_duplicate_points(coords)
    return coords

    # # Spline interpolation
    # try:
    #     tck, _ = splprep([coords[:, 0], coords[:, 1]], s=1e-6)
    #     u_new = np.linspace(0, 1, n_points)
    #     x_new, y_new = splev(u_new, tck)
    #     return np.column_stack((x_new, y_new))
    # except Exception:
    #     # Fallback: just return the original (deduplicated)
    #     return coords

def write_airfoil(coords, out_path, header):
    with open(out_path, 'w') as f:
        f.write(header)
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")

def clean_airfoil_file(input_path, output_path):
    header, coords = read_airfoil(input_path)
    coords = remove_duplicate_points(coords)
    upper, lower = split_surfaces(coords)
    upper, lower = enforce_trailing_edge(upper, lower)
    # visualize_coords(upper, title="Upper Surface")
    # visualize_coords(lower, title="Lower Surface")
    cleaned = merge_and_resample(upper, lower)
    write_airfoil(cleaned, output_path, header)

def process_airfoils(database, airfoil=None, process_all=False):
    input_dir = database
    output_dir = database + "_cleaned"
    os.makedirs(output_dir, exist_ok=True)

    if process_all:
        files = [f for f in os.listdir(input_dir) if f.endswith(".dat")]
    else:
        if airfoil is None:
            raise ValueError("You must specify --airfoil or use --all")
        files = [airfoil + ".dat"]

    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        print(f"Cleaning {file} -> {output_path}")
        clean_airfoil_file(input_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Clean airfoil DAT files for meshing")
    parser.add_argument("--database", default='airfoil_database', help="Folder containing .dat airfoil files")
    parser.add_argument("--airfoil", default='clarky_rot-1', help="Airfoil name (without .dat extension)")
    parser.add_argument("--all", action="store_true", help="Process all .dat files in the database folder")
    args = parser.parse_args()

    process_airfoils(args.database, args.airfoil, args.all)

if __name__ == "__main__":
    main()
