import subprocess
import os
import random
import argparse
import sys
from pathlib import Path
import numpy as np
import shutil


class MeshCreationError(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate mesh files from airfoil data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --all                    # Process all airfoils
  %(prog)s --airfoil NACA0012       # Process specific airfoil
  %(prog)s --samples 20             # Generate 20 random samples
  %(prog)s --seed 12345             # Use specific random seed
  %(prog)s --list                   # List available airfoils
'''
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true',
                      help='Process all airfoils in database')
    group.add_argument('--airfoil', type=str,
                      help='Process specific airfoil file')
    group.add_argument('--samples', type=int, default=10,
                      help='Number of random samples to generate (default: 10)')
    group.add_argument('--list', action='store_true',
                      help='List all available airfoils')
    
    parser.add_argument('--database', type=str, default='./airfoil_database/',
                       help='Path to airfoil database directory (default: ./airfoil_database/)')
    parser.add_argument('--output', type=str, default='./re2_files/',
                       help='Output directory for .re2 files (default: ./re2_files/)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible results')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup directories and random seed"""
    os.makedirs(args.output, exist_ok=True)
    
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 2**32 - 1)
    
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")
    return seed

def genMesh(airfoilFile, output_dir):
    airfoilName = Path(airfoilFile).stem
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[-1]))<1e-6:
        ar = ar[:-1]

    if np.abs(ar[0][1]+ar[-1][1]) < 1e-6:
        ar = ar[:-1]
        ar[0][1] = 0.

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += f"Point({pointIndex}) = {{ {ar[n][0]}, {ar[n][1]}, 0.00000000, 0.005}};\n"
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", output)
                line = line.replace("LAST_POINT_INDEX", str(pointIndex-1))
                outFile.write(line)

    try:
        subprocess.run(
            ["gmsh", "airfoil.geo", "-2", "-format", "msh2", "-order", "2", "-o", "airfoil.msh"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=True
        )
    except subprocess.TimeoutExpired:
        raise MeshCreationError("GMSH mesh generation timed out after 30 seconds")
    except subprocess.CalledProcessError:
        raise MeshCreationError("GMSH failed to create mesh")


    print("GMSH complete")

    responses = ["2", "airfoil", "0", "0", airfoilName]
    input_data = "\n".join(responses) + "\n"
    process = subprocess.Popen(
        ["./gmsh2nek"],
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE,  
        stderr=subprocess.PIPE,  
        text=True
    )
    stdout, stderr = process.communicate(input=input_data)
    print("NEK complete")
    try:
        shutil.move(f"{airfoilName}.re2", output_dir)  
    except shutil.Error:
        if os.path.exists(f"{airfoilName}.re2"):
            os.remove(f"{airfoilName}.re2")

    return 0

def get_airfoil_files(database_path):
    """Get list of airfoil files from database"""
    if not os.path.exists(database_path):
        print(f"Error: Airfoil database directory '{database_path}' not found")
        sys.exit(1)
    
    files = [f for f in os.listdir(database_path) if f.endswith('.dat')]
    files.sort()
    
    if len(files) == 0:
        print(f"Error: No airfoil files found in {database_path}")
        sys.exit(1)
    
    return files


def list_airfoils(database_path):
    """List all available airfoils"""
    files = get_airfoil_files(database_path)
    print(f"Found {len(files)} airfoil files in {database_path}:")
    for i, file in enumerate(files, 1):
        print(f"  {i:4d}. {file.split('.')[0]}")


def process_all_airfoils(database_path, output_dir):
    """Process all airfoils in database"""
    files = get_airfoil_files(database_path)
    print(f"Processing all {len(files)} airfoils...")
    
    failed_files = []
    for i, file in enumerate(files, 1):
        print(f"Processing {i}/{len(files)}: {file}")
        try:
            genMesh(os.path.join(database_path, file), output_dir)
            print(f"  ✓ Success")
        except MeshCreationError as e:
            print(f"  ✗ Failed: {e}")
            failed_files.append(file)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed_files.append(file)
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for file in failed_files:
            print(f"  - {file}")
    else:
        print(f"\n✓ Successfully processed all {len(files)} airfoils")


def process_specific_airfoil(airfoil_name, database_path, output_dir):
    """Process a specific airfoil"""
    airfoil_path = os.path.join(database_path, airfoil_name + ".dat")
    
    if not os.path.exists(airfoil_path):
        print(f"Error: Airfoil file '{airfoil_name}' not found in {database_path}")
        available_files = get_airfoil_files(database_path)
        print(f"Available files: {', '.join(available_files[:5])}{'...' if len(available_files) > 5 else ''}")
        sys.exit(1)
    
    print(f"Processing airfoil: {airfoil_name}")
    try:
        genMesh(airfoil_path, output_dir)
        print("✓ Success")
    except MeshCreationError as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)


def process_random_samples(num_samples, database_path, output_dir):
    """Process random samples from database"""
    files = get_airfoil_files(database_path)
    
    print(f"Generating {num_samples} random samples from {len(files)} available airfoils...")
    
    failed_count = 0
    for n in range(num_samples):
        print(f"Sample {n+1}/{num_samples}:")
        
        file_index = np.random.randint(0, len(files))
        selected_file = files[file_index]
        
        print(f"  Using: {selected_file}")
        
        try:
            genMesh(os.path.join(database_path, selected_file), output_dir)
            print("  ✓ Success")
        except MeshCreationError as e:
            print(f"  ✗ Failed: {e}")
            failed_count += 1
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed_count += 1
    
    success_count = num_samples - failed_count
    print(f"\nCompleted: {success_count}/{num_samples} successful, {failed_count} failed")


def main():
    args = parse_arguments()
    
    if args.list:
        list_airfoils(args.database)
        return
    
    setup_environment(args)
    
    if args.all:
        process_all_airfoils(args.database, args.output)
    elif args.airfoil:
        process_specific_airfoil(args.airfoil, args.database, args.output)
    else:  # args.samples
        process_random_samples(args.samples, args.database, args.output)


if __name__ == "__main__":
    main()
