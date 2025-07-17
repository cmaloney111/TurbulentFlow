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
    "sd2030"
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

# PBS configuration
PBS_TEMPLATE = """#!/bin/bash
#PBS -N {job_name}
#PBS -o {output_file}
#PBS -q debug
#PBS -l select=1:ncpus=2:mpiprocs=2
#PBS -l walltime={time_limit}
#PBS -A UncertaintyDL
#PBS -l filesystems=home:eagle

module purge
module load PrgEnv-gnu
module load cray-mpich

cd {work_dir}

# Run the simulation
{commands}
"""







def create_single_pbs(job_infos, output_path="run_all.pbs", time_limit="04:00:00"):
    """
    job_infos is a list of dicts, each with keys:
      'rotated_name', 'reynolds', 'work_dir'
    This writes one big PBS script that, for each case,
    cds into its work_dir, runs genmap, does the initial
    and restart nekmpi, all in series.
    """
    all_cmds = []
    for info in job_infos:
        name = info["rotated_name"]
        wd   = info["work_dir"]
        ry   = info["reynolds"]
        # 1) mesh + initial run
        all_cmds.append(f"cd {wd}")
        all_cmds.append("genmap < genmap_input.txt")
        all_cmds.append(f"echo 'Initial run for {name}'")
        all_cmds.append(f"nekmpi {name} 2")
        # 2) inline Python to edit the .par for restart
        all_cmds.append(f"""python3 << 'EOF'
import re
par_file = '{name}.par'
with open(par_file, 'r') as f:
    content = f.read()
content = re.sub(r'-10000\\.0', '-{ry}', content)
content = re.sub(r'-10000',     '-{ry}', content)
content = re.sub(r'#startFrom = rans0.f00001',
                 'startFrom = {name}0.f00001',
                 content)
content = re.sub(r'#timeStepper = BDF2',
                 'timeStepper = BDF2',
                 content)
content = re.sub(r'#extrapolation = OIFS',
                 'extrapolation = OIFS',
                 content)
content = re.sub(r'#targetCFL = 3.5\\.',
                 'targetCFL = 3.5.',
                 content)
content = re.sub(r'numsteps = 2',
                 'numsteps = 5',
                 content)
content = re.sub(r'writeInterval = 2',
                 'writeInterval = 5',
                 content)
with open(par_file, 'w') as f:
    f.write(content)
EOF""")
        # 3) restart run
        all_cmds.append(f"echo 'Restart run for {name}'")
        all_cmds.append(f"nekmpi {name} 2")
        all_cmds.append("")  # blank line to separate cases

    script = PBS_TEMPLATE.format(
        job_name    = "rans_all",
        output_file = "rans_all.out",
        time_limit  = time_limit,
        work_dir    = ".",
        commands    = "\n".join(all_cmds)
    )
    with open(output_path, "w") as f:
        f.write(script)
    print(f"→ single PBS script written to ./{output_path}")
    return output_path











import subprocess






def run_command(cmd, cwd=None, input_text=None, check=True):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            input=input_text
        )

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

def create_pbs_job(job_info):
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
content = re.sub(r'#startFrom = rans0.f00001', 'startFrom = {job_info['rotated_name']}0.f00001', content)
content = re.sub(r'#timeStepper = BDF2', 'timeStepper = BDF2', content)
content = re.sub(r'#extrapolation = OIFS', 'extrapolation = OIFS', content)
content = re.sub(r'#targetCFL = 3.5.', 'targetCFL = 3.5.', content)
content = re.sub(r'numsteps = 2', 'numsteps = 5', content)
content = re.sub(r'writeInterval = 2', 'writeInterval = 5', content)

with open(par_file, 'w') as f:
    f.write(content)
EOF
"""
    commands.append(update_script)
    
    # Restart run
    commands.append(f"echo 'Starting restart run at Re={job_info['reynolds']}...'")
    commands.append(f"nekmpi {job_info['rotated_name']} 2")
    
    # Create job script
    job_script = PBS_TEMPLATE.format(
        job_name=job_name,
        output_file=output_file,
        time_limit="00:15:00",
        work_dir=job_info['work_dir'],
        commands='\n'.join(commands)
    )
    
    # Save job script
    job_file = SLURM_JOBS_DIR / f"{job_name}.sbatch"
    with open(job_file, 'w') as f:
        f.write(job_script)
    
    return job_file

    




def prepare_all_simulations(airfoil_name, df):
    """Rotate, convert, and prepare directories; return a list of job_info dicts."""
    print(f"\n{'='*60}")
    print(f"Preparing simulations for: {airfoil_name}")
    print(f"{'='*60}")

    reynolds_to_angles = get_angles_and_reynolds(airfoil_name, df)
    job_infos = []

    # Phase 1: rotate and convert
    print("\nPhase 1: Rotating airfoils and converting to RE2...")
    for reynolds, angles in reynolds_to_angles.items():
        for angle in angles:
            print(f"  Preparing {airfoil_name} at {angle}° for Re={reynolds}")
            rotated_name = rotate_airfoil(airfoil_name, angle)
            convert_to_re2(rotated_name)

    # Phase 2: make run directories and collect their info
    print("\nPhase 2: Preparing simulation directories...")
    for reynolds, angles in reynolds_to_angles.items():
        for angle in angles:
            rans_dir = create_rans_directory(airfoil_name, angle, reynolds)
            rotated_name = prepare_simulation_files(rans_dir, airfoil_name, angle, reynolds)

            job_infos.append({
                'airfoil':      airfoil_name,
                'angle':        angle,
                'reynolds':     reynolds,
                'rotated_name': rotated_name,
                'work_dir':     str(rans_dir)
            })
            print(f"  Prepared directory: {rans_dir.name}")

    return job_infos










def submit_jobs(job_files):
    """Submit PBS jobs with qsub (no interactive shell)."""
    job_ids = []

    for job_file in job_files:
        cmd = [
            "qsub",
            "-q", "debug",
            "-A", "UncertaintyDL",
            "-l", "select=1:ncpus=2:mpiprocs=2",
            "-l", "walltime=01:00:00",
            "-l", "filesystems=home",
            str(job_file)
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=str(job_file.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
)

        if result.returncode == 0:
            job_id = result.stdout.strip().split('.')[0]
            job_ids.append(job_id)
            print(f"Submitted {job_file.name} → {job_id}")
        else:
            print(f"Failed to submit {job_file.name}")
            print(result.stderr.strip())

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
    






#    all_job_files = []
    
#    for airfoil_name in AIRFOILS_TO_PROCESS:
#        if check_airfoil_exists(airfoil_name, df):
#            job_files = prepare_all_simulations(airfoil_name, df)
#            all_job_files.extend(job_files)

#        else:
#            print(f"\nSkipping {airfoil_name}: Not found in both CSV and database")





    all_job_infos = []
    for af in AIRFOILS_TO_PROCESS:
        infos = prepare_all_simulations(af, df)    # have that return a list of job_info dicts
        all_job_infos.extend(infos)

    # now build the single submit script
    pbs_file = create_single_pbs(all_job_infos,
                                 output_path="run_all.pbs",
                                 time_limit="01:00:00")

    print("\nAll setup complete.")
    print("To launch, just:")
    print(f"    qsub {pbs_file}")
    sys.exit(0)










    
    if all_job_files:
        print(f"\n{'='*60}")
        print(f"Created {len(all_job_files)} job scripts")
        print(f"{'='*60}")
        
        # Ask user about job submission
        response = input("\nSubmit all jobs to SLURM? (y/n): ").lower()
        if response == 'y':
            print("\nSubmitting jobs...")
            job_ids = submit_jobs(all_job_files)  # No dependencies for max parallelism
            
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
