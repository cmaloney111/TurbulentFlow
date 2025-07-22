#!/usr/bin/env python3
"""
Parallelized RANS Simulation Automation Script for Polaris
Creates PBS batch jobs for parallel execution of RANS simulations
"""

import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json

# Configuration - List of airfoils to process
AIRFOILS_TO_PROCESS = [
    "s4180",
    #"sd2030"
]

# Directory paths
SCRIPT_DIR = Path(__file__).parent.absolute()
GENAIRFOILS_DIR = SCRIPT_DIR / "GenAirfoils"
AIRFOIL_DB_DIR = GENAIRFOILS_DIR / "airfoil_database"
RE2_FILES_DIR = GENAIRFOILS_DIR / "re2_files"
RANS_BASE_DIR = SCRIPT_DIR / "rans_base"
RANS_RUNS_DIR = SCRIPT_DIR / "rans_runs"
CSV_FILE = SCRIPT_DIR / "training_data_stec8.csv"
PBS_JOBS_DIR = SCRIPT_DIR / "pbs_jobs"

# PBS configuration for Polaris - Single large job
PBS_TEMPLATE = """#!/bin/bash
#PBS -N rans_batch_{batch_id}
#PBS -o rans_batch_{batch_id}.o
#PBS -q debug
#PBS -A UncertaintyDL
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime="1:00:00"
#PBS -l filesystems=home

# Load modules (adjust these based on your Polaris setup)
#module purge
#module load PrgEnv-gnu
#module load cray-mpich
#module load craype-x86-milan
#module load libfabric

# Set environment variables
#export MPICH_OFI_VERBOSE=1
#export MPICH_GPU_SUPPORT_ENABLED=0

# Verify modules are loaded
#echo "Loaded modules:"
#module list

# Check MPI installation
#echo "MPI compiler info:"
#which mpicc
#which mpiexec
#ldd ./nek5000 | grep -i mpi
# Function to run a single simulation
run_simulation() {{
    local work_dir=$1
    local rotated_name=$2
    local reynolds=$3

    echo "=============================================="
    echo "Starting simulation: $rotated_name at Re=$reynolds"
    echo "Working directory: $work_dir"
    echo "=============================================="

    cd "$work_dir"

    # Run genmap
    echo "Running genmap..."
    genmap < genmap_input.txt

    if [ $? -ne 0 ]; then
        echo "ERROR: genmap failed for $rotated_name"
        return 1
    fi

    # Initial run
    echo "Starting initial run..."
    nekmpi $rotated_name 2
    if [ $? -ne 0 ]; then
        echo "ERROR: Initial run failed for $rotated_name"
        return 1
    fi

    # Update par file for restart
    echo "Updating par file for restart..."
    python3 << EOF
import re

par_file = '${{rotated_name}}.par'
with open(par_file, 'r') as f:
    content = f.read()

# Update for restart
content = re.sub(r'-10000.0', '-${{reynolds}}', content)
content = re.sub(r'-10000', '-${{reynolds}}', content)
content = re.sub(r'#startFrom = rans0.f00001', 'startFrom = ${{rotated_name}}0.f00001', content)
content = re.sub(r'#timeStepper = BDF2', 'timeStepper = BDF2', content)
content = re.sub(r'#extrapolation = OIFS', 'extrapolation = OIFS', content)
content = re.sub(r'#targetCFL = 3.5.', 'targetCFL = 3.5.', content)
content = re.sub(r'numsteps = 2', 'numsteps = 5', content)
content = re.sub(r'writeInterval = 2', 'writeInterval = 5', content)

with open(par_file, 'w') as f:
    f.write(content)
EOF

    # Restart run
    echo "Starting restart run at Re=$reynolds..."
    nekmpi $rotated_name 2
    if [ $? -ne 0 ]; then
        echo "ERROR: Restart run failed for $rotated_name"
        return 1
    fi

    echo "Simulation completed successfully: $rotated_name at Re=$reynolds"
    echo ""

    return 0
}}

# Run all simulations
{simulation_calls}

echo "=============================================="
echo "Batch job completed"
echo "=============================================="
"""

def run_command(cmd, cwd=None, input_text=None, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, cwd=cwd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=input_text)

        if check and result.returncode != 0:
            print(f"Error running command: {result.stderr}")
            sys.exit(1)
        return result
    except Exception as e:
        print(f"Exception running command: {e}")
        if check:
            sys.exit(1)
        return None

def check_airfoil_exists(airfoil_name, df):
    """Check if airfoil exists in both CSV (uppercase) and database (lowercase)."""
    csv_exists = airfoil_name.upper() in df['airfoil_name'].values
    dat_file = AIRFOIL_DB_DIR / f"{airfoil_name.lower()}.dat"
    db_exists = dat_file.exists()
    return csv_exists and db_exists

def get_angles_and_reynolds(airfoil_name, df):
    """Get angles of attack for each Reynolds number for an airfoil."""
    airfoil_data = df[df['airfoil_name'] == airfoil_name.upper()]
    reynolds_to_angles = {}

    for reynolds in airfoil_data['reynolds_number'].unique():
        reynolds_data = airfoil_data[airfoil_data['reynolds_number'] == reynolds]
        angles_rad = reynolds_data['angle_of_attack'].unique()
        angles_deg = [int(np.floor(angle)) for angle in angles_rad]
        reynolds_to_angles[reynolds] = sorted(angles_deg)

    return reynolds_to_angles

def rotate_airfoil(airfoil_name, angle):
    """Rotate an airfoil using rotate_dat.py."""
    if angle == 0:
        return airfoil_name

    # Check if rotated file already exists
    rotated_name = f"{airfoil_name}_rot{angle}"
    rotated_file = AIRFOIL_DB_DIR / f"{rotated_name}.dat"

    if not rotated_file.exists():
        cmd = [
            "python",
            str(GENAIRFOILS_DIR / "rotate_dat.py"),
            "--airfoil", airfoil_name,
            str(angle)
        ]
        run_command(cmd, cwd=GENAIRFOILS_DIR)

    return rotated_name

def convert_to_re2(airfoil_name):
    """Convert airfoil to RE2 format using test_gen.py."""
    # Check if RE2 file already exists
    re2_file = RE2_FILES_DIR / f"{airfoil_name}.re2"

    if not re2_file.exists():
        cmd = [
            "python",
            str(GENAIRFOILS_DIR / "test_gen.py"),
            "--airfoil", airfoil_name
        ]
        run_command(cmd, cwd=GENAIRFOILS_DIR)

def create_rans_directory(airfoil_name, angle, reynolds):
    """Create directory structure for RANS run."""
    angle_dir = str(angle).replace("-", "neg")
    rans_dir = RANS_RUNS_DIR / airfoil_name / f"Re{int(reynolds)}" / angle_dir
    rans_dir.mkdir(parents=True, exist_ok=True)
    return rans_dir

def prepare_simulation_files(rans_dir, airfoil_name, angle, reynolds):
    """Prepare all files needed for simulation."""
    # Copy all files from rans_base
    for file in RANS_BASE_DIR.iterdir():
        if file.is_file():
            shutil.copy2(file, rans_dir)

    # Determine the RE2 filename
    if angle == 0:
        re2_name = f"{airfoil_name}.re2"
        rotated_name = airfoil_name
    else:
        rotated_name = f"{airfoil_name}_rot{angle}"
        re2_name = f"{rotated_name}.re2"

    # Copy RE2 file
    re2_source = RE2_FILES_DIR / re2_name
    shutil.copy2(re2_source, rans_dir)

    # Create genmap input file
    genmap_input_file = rans_dir / "genmap_input.txt"
    with open(genmap_input_file, 'w') as f:
        f.write(f"{rotated_name}\n0.05\n")

    # Prepare initial par file
    old_par = rans_dir / "rans.par"
    new_par = rans_dir / f"{rotated_name}.par"
    shutil.copy2(old_par, new_par)
    old_par.unlink()

    return rotated_name

def create_batch_job(all_job_infos):
    """Create a single PBS job script that runs all simulations."""
    batch_id = "1"

    # Create simulation calls
    simulation_calls = []
    for job_info in all_job_infos:
        call = f"run_simulation '{job_info['work_dir']}' '{job_info['rotated_name']}' '{job_info['reynolds']}'"
        simulation_calls.append(call)

    # Create job script
    job_script = PBS_TEMPLATE.format(
        batch_id=batch_id,
        simulation_calls='\n'.join(simulation_calls)
    )

    # Save job script
    job_file = PBS_JOBS_DIR / f"rans_batch_{batch_id}.pbs"
    with open(job_file, 'w') as f:
        f.write(job_script)

    return [job_file]

def prepare_all_simulations(airfoil_name, df):
    """Prepare all simulations for an airfoil."""
    print(f"\n{'='*60}")
    print(f"Preparing simulations for: {airfoil_name}")
    print(f"{'='*60}")

    reynolds_to_angles = get_angles_and_reynolds(airfoil_name, df)
    job_infos = []

    # First pass: rotate airfoils and convert to RE2
    print("\nPhase 1: Rotating airfoils and converting to RE2...")
    for reynolds, angles in reynolds_to_angles.items():
        for angle in angles:
            print(f"  Preparing {airfoil_name} at {angle}° for Re={reynolds}")
            rotated_name = rotate_airfoil(airfoil_name, angle)
            convert_to_re2(rotated_name)

    # Second pass: prepare simulation directories
    print("\nPhase 2: Preparing simulation directories...")
    for reynolds, angles in reynolds_to_angles.items():
        for angle in angles:
            rans_dir = create_rans_directory(airfoil_name, angle, reynolds)
            rotated_name = prepare_simulation_files(rans_dir, airfoil_name, angle, reynolds)

            job_info = {
                'airfoil': airfoil_name,
                'angle': angle,
                'reynolds': reynolds,
                'rotated_name': rotated_name,
                'work_dir': str(rans_dir)
            }

            job_infos.append(job_info)
            print(f"  Prepared simulation: {rotated_name} at Re={reynolds}")

    return job_infos

def submit_jobs(job_files):
    """Submit PBS jobs."""
    job_ids = []

    for job_file in job_files:
        cmd = ["qsub", str(job_file)]

        result = run_command(cmd, check=False)

        if result and result.returncode == 0:
            # Extract job ID from output
            job_id = result.stdout.strip()
            job_ids.append(job_id)
            print(f"Submitted job {job_file.name} with ID: {job_id}")
        else:
            print(f"Failed to submit job {job_file.name}")
            if result:
                print(f"Error: {result.stderr}")

    return job_ids

def main():
    """Main function to orchestrate the workflow."""
    print("Parallelized RANS Simulation Automation Script for Polaris")
    print("========================================================")

    # Create necessary directories
    RANS_RUNS_DIR.mkdir(exist_ok=True)
    PBS_JOBS_DIR.mkdir(exist_ok=True)

    # Load CSV data
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    all_job_infos = []

    # Prepare all simulations
    for airfoil_name in AIRFOILS_TO_PROCESS:
        if check_airfoil_exists(airfoil_name, df):
            job_infos = prepare_all_simulations(airfoil_name, df)
            all_job_infos.extend(job_infos)
        else:
            print(f"\nSkipping {airfoil_name}: Not found in both CSV and database")

    if all_job_infos:
        print(f"\n{'='*60}")
        print(f"Prepared {len(all_job_infos)} simulations")
        print(f"Creating single batch job for all simulations...")
        print(f"{'='*60}")

        # Create single batch job
        batch_files = create_batch_job(all_job_infos)
        print(f"\nCreated 1 batch job with {len(all_job_infos)} simulations")

        # Ask user about job submission
        response = input("\nSubmit batch job to PBS? (y/n): ").lower()
        if response == 'y':
            print("\nSubmitting job...")
            job_ids = submit_jobs(batch_files)

            print(f"\n{'='*60}")
            print(f"Submitted batch job successfully!")
            print("Monitor progress with: qstat -u $USER")
            print("Check outputs in: rans_batch_1.o*")
            print(f"{'='*60}")
        else:
            print("\nBatch job script created but not submitted.")
            print(f"To submit manually, run: qsub {batch_files[0]}")
            print(f"Batch script is in: {PBS_JOBS_DIR}")
    else:
        print("\nNo simulations to run.")

if __name__ == "__main__":
    main()
