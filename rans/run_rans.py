#!/usr/bin/env python3
"""
Parallelized RANS Simulation Automation Script
Creates SLURM batch jobs for parallel execution of RANS simulations
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
    "aquila",
    #"clark-y",
    #"dae51",
    #"df101",
    #"df102",
    #"e193",
    #"fx60-100",
    #"j5012",
    #"mb253515",
]

# Directory paths
SCRIPT_DIR = Path(__file__).parent.absolute()
GENAIRFOILS_DIR = SCRIPT_DIR / "GenAirfoils"
AIRFOIL_DB_DIR = GENAIRFOILS_DIR / "airfoil_database"
RE2_FILES_DIR = GENAIRFOILS_DIR / "re2_files"
RANS_BASE_DIR = SCRIPT_DIR / "rans_base"
RANS_RUNS_DIR = SCRIPT_DIR / "rans_runs"
CSV_FILE = SCRIPT_DIR / "training_data_stec8.csv"
SLURM_JOBS_DIR = SCRIPT_DIR / "slurm_jobs"

# SLURM configuration
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --output="{output_file}"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8G
#SBATCH --account=slo102
#SBATCH --export=ALL
#SBATCH -t {time_limit}

module purge
module load slurm
module load cpu/0.15.4
module load intel/19.1.1.217
module load openmpi/3.1.6

cd {work_dir}

# Run the simulation
{commands}
"""

def run_command(cmd, cwd=None, input_text=None, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, input=input_text)
        
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
        dest = rans_dir / file.name
        if file.is_file():
            if file.name == "nek5000":
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                # Create symlink
                os.symlink(file.resolve(), dest)
            else:
                shutil.copy2(file, dest) 
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

def create_slurm_job(job_info):
    """Create a SLURM job script for a specific simulation."""
    job_name = f"{job_info['airfoil']}_Re{int(job_info['reynolds'])}_a{job_info['angle']}"
    output_file = SLURM_JOBS_DIR / f"{job_name}.%j.out"
    
    # Build command sequence
    commands = []
    
    # Run genmap
    commands.append(f"genmap < genmap_input.txt")
    
    # Initial run
    commands.append(f"echo 'Starting initial run...'")
    commands.append(f"nekmpi {job_info['rotated_name']} 2")
    
    # Update par file for restart
    update_script = f"""
python3 << 'EOF'
import re

par_file = '{job_info['rotated_name']}.par'
with open(par_file, 'r') as f:
    content = f.read()

# Update for restart
content = re.sub(r'-10000\.0', '-{job_info['reynolds']}', content)
content = re.sub(r'-10000', '-{job_info['reynolds']}', content)
content = re.sub(r'#startFrom = rans0.f00001', 'startFrom = {job_info['rotated_name']}0.f00002', content)
content = re.sub(r'#timeStepper = BDF2', 'timeStepper = BDF2', content)
content = re.sub(r'#extrapolation = OIFS', 'extrapolation = OIFS', content)
content = re.sub(r'#targetCFL = 3.5.', 'targetCFL = 3.5.', content)
content = re.sub(r'dt = 1.0e-7', 'dt = 1.0e-5', content)
content = re.sub(r'numsteps = 10', 'numsteps = 100', content)
content = re.sub(r'writeInterval = 10', 'writeInterval = 100', content)

with open(par_file, 'w') as f:
    f.write(content)
EOF
"""
    commands.append(update_script)
    
    # Restart run
    commands.append(f"echo 'Starting restart run at Re={job_info['reynolds']}...'")
    commands.append(f"nekmpi {job_info['rotated_name']} 2")
    
    # Create job script
    job_script = SLURM_TEMPLATE.format(
        job_name=job_name,
        output_file=output_file,
        time_limit="00:30:00",
        work_dir=job_info['work_dir'],
        commands='\n'.join(commands)
    )
    
    # Save job script
    job_file = SLURM_JOBS_DIR / f"{job_name}.sbatch"
    with open(job_file, 'w') as f:
        f.write(job_script)
    
    return job_file

def prepare_all_simulations(airfoil_name, df):
    """Prepare all simulations for an airfoil."""
    print(f"\n{'='*60}")
    print(f"Preparing simulations for: {airfoil_name}")
    print(f"{'='*60}")
    
    reynolds_to_angles = get_angles_and_reynolds(airfoil_name, df)
    job_files = []
    
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
            
            job_file = create_slurm_job(job_info)
            job_files.append(job_file)
            print(f"  Created job script: {job_file.name}")
    return job_files

def submit_jobs(job_files, dependency_type='singleton'):
    """Submit SLURM jobs with optional dependencies."""
    job_ids = []
    
    for i, job_file in enumerate(job_files):
        cmd = ["sbatch"]
        
        # Add dependency if not the first job and dependency type is set
        if i > 0 and dependency_type:
            if dependency_type == 'singleton':
                # Jobs with same name run sequentially
                cmd.extend(["--dependency=singleton"])
            elif dependency_type == 'afterok' and job_ids:
                # Wait for previous job to complete successfully
                cmd.extend([f"--dependency=afterok:{job_ids[-1]}"])
        
        cmd.append(str(job_file))
        
        result = run_command(cmd, check=False)
        
        if result and result.returncode == 0:
            # Extract job ID from output
            output = result.stdout.strip()
            job_id = output.split()[-1]
            job_ids.append(job_id)
            print(f"Submitted job {job_file.name} with ID: {job_id}")
        else:
            print(f"Failed to submit job {job_file.name}: {result.stderr}")
    
    return job_ids

def main():
    """Main function to orchestrate the workflow."""
    print("Parallelized RANS Simulation Automation Script")
    print("=============================================")
    
    # Create necessary directories
    RANS_RUNS_DIR.mkdir(exist_ok=True)
    SLURM_JOBS_DIR.mkdir(exist_ok=True)
    
    # Load CSV data
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    all_job_files = []
    
    # Prepare all simulations
    for airfoil_name in AIRFOILS_TO_PROCESS:
        if check_airfoil_exists(airfoil_name, df):
            job_files = prepare_all_simulations(airfoil_name, df)
            all_job_files.extend(job_files)
        else:
            print(f"\nSkipping {airfoil_name}: Not found in both CSV and database")
    
    if all_job_files:
        print(f"\n{'='*60}")
        print(f"Created {len(all_job_files)} job scripts")
        print(f"{'='*60}")
        
        # Ask user about job submission
        response = input("\nSubmit all jobs to SLURM? (y/n): ").lower()
        if response == 'y':
            print("\nSubmitting jobs...")
            job_ids = submit_jobs(all_job_files, dependency_type=None)  # No dependencies for max parallelism
            
            print(f"\n{'='*60}")
            print(f"Submitted {len(job_ids)} jobs successfully!")
            print("Monitor progress with: squeue -u $USER")
            print("Check outputs in: slurm_jobs/*.out")
            print(f"{'='*60}")
        else:
            print("\nJob scripts created but not submitted.")
            print(f"To submit manually, run: sbatch <job_script>")
            print(f"Job scripts are in: {SLURM_JOBS_DIR}")
    else:
        print("\nNo simulations to run.")

if __name__ == "__main__":
    main()
