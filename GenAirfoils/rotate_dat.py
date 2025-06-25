#!/usr/bin/env python3
# 
#
# How to use:
#
# default database (input) is './airfoil_database/'
# default output directory is './airfoil_database/'
#
# List all available airfoils   
# python3 rotate_dat.py --list
#
# # Rotate all airfoils by 15 degrees
# python3 rotate_dat.py --all 15
#
# # Rotate a specific airfoil by 30 degrees
# python3 rotate_dat.py --airfoil a18 30
#
# # Use custom directories (for specific output into ./airfoil_rot_database/)
# python3 rotate_dat.py --all 45 --database ./my_airfoils/ --output ./airfoil_rot_database/

import argparse
import os
import sys
import numpy as np
from pathlib import Path


def rotate_dat(inpath, outpath, angle_deg):
    """Read 2D points from inpath, rotate by angle_deg, write to outpath."""
    # Load all points (skip first header line)
    pts = np.loadtxt(inpath, skiprows=1)
    # Drop duplicate endpoint if present
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    # Build rotation matrix
    θ = np.deg2rad(-angle_deg)
    c, s = np.cos(θ), np.sin(θ)
    rot_pts = np.column_stack([pts[:,0]*c - pts[:,1]*s,
                               pts[:,0]*s + pts[:,1]*c])

    # Preserve the original header
    with open(inpath, 'r') as f:
        header = f.readline()
    with open(outpath, 'w') as f:
        f.write(header)
        np.savetxt(f, rot_pts, fmt="%.6f %.6f")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Rotate airfoil .dat geometry files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --all 15                 # Rotate all airfoils by 15 degrees
  %(prog)s --airfoil NACA0012 30    # Rotate specific airfoil by 30 degrees
  %(prog)s --list                   # List available airfoils
'''
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', type=float, metavar='ANGLE',
                      help='Rotate all airfoils in database by ANGLE degrees')
    group.add_argument('--airfoil', type=str, metavar='NAME',
                      help='Rotate specific airfoil file')
    group.add_argument('--list', action='store_true',
                      help='List all available airfoils')
    
    parser.add_argument('angle', type=float, nargs='?',
                       help='Rotation angle in degrees (positive is counter-clockwise)')
    parser.add_argument('--database', type=str, default='./airfoil_database/',
                       help='Path to airfoil database directory (default: ./airfoil_database/)')
    parser.add_argument('--output', type=str, default='./airfoil_database/',
                       help='Output directory for rotated .dat files (default: ./airfoil_database/)')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup output directory"""
    os.makedirs(args.output, exist_ok=True)


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


def process_all_airfoils(database_path, output_dir, angle):
    """Process all airfoils in database"""
    files = get_airfoil_files(database_path)
    print(f"Rotating all {len(files)} airfoils by {angle} degrees...")
    
    failed_files = []
    for i, file in enumerate(files, 1):
        print(f"Processing {i}/{len(files)}: {file}")
        try:
            input_path = os.path.join(database_path, file)
            base_name = Path(file).stem
            output_filename = f"{base_name}_rot{int(angle)}.dat"
            output_path = os.path.join(output_dir, output_filename)
            
            rotate_dat(input_path, output_path, angle)
            print(f"  ✓ Success -> {output_filename}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_files.append(file)
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for file in failed_files:
            print(f"  - {file}")
    else:
        print(f"\n✓ Successfully processed all {len(files)} airfoils")


def process_specific_airfoil(airfoil_name, database_path, output_dir, angle):
    """Process a specific airfoil"""
    airfoil_path = os.path.join(database_path, airfoil_name + ".dat")
    
    if not os.path.exists(airfoil_path):
        print(f"Error: Airfoil file '{airfoil_name}' not found in {database_path}")
        available_files = get_airfoil_files(database_path)
        available_names = [Path(f).stem for f in available_files]
        print(f"Available airfoils: {', '.join(available_names[:5])}{'...' if len(available_names) > 5 else ''}")
        sys.exit(1)
    
    print(f"Rotating airfoil: {airfoil_name} by {angle} degrees")
    try:
        output_filename = f"{airfoil_name}_rot{int(angle)}.dat"
        output_path = os.path.join(output_dir, output_filename)
        
        rotate_dat(airfoil_path, output_path, angle)
        print(f"✓ Success -> {output_filename}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)


def main():
    args = parse_arguments()
    
    if args.list:
        list_airfoils(args.database)
        return
    
    # Validate angle argument
    if args.all is not None:
        angle = args.all
    elif args.airfoil is not None:
        if args.angle is None:
            print("Error: Angle is required when using --airfoil option")
            sys.exit(1)
        angle = args.angle
    else:
        print("Error: Either --all, --airfoil, or --list must be specified")
        sys.exit(1)
    
    setup_environment(args)
    
    if args.all is not None:
        process_all_airfoils(args.database, args.output, angle)
    elif args.airfoil:
        process_specific_airfoil(args.airfoil, args.database, args.output, angle)


if __name__ == '__main__':
    main()