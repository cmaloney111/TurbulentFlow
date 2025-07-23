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
import math

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

# PBS configuration for Polaris
# Since nekmpi uses 2 tasks per node, we need to account for this
# Polaris has 64 cores per node
CORES_PER_NODE = 256
TASKS_PER_SIMULATION = 2
MAX_NODES = 184  # Maximum for workq-route
DEFAULT_NODES = 8  # Default number of nodes to request
DEFAULT_WALLTIME = "4:00:00"  # Default walltime

PBS_TEMPLATE = """#!/bin/bash
#PBS -N rans_batch_{batch_id}
#PBS -o {pbs_jobs_dir}/rans_batch_{batch_id}.o
#PBS -e {pbs_jobs_dir}/rans_batch_{batch_id}.e
#PBS -q {queue}
#PBS -A UncertaintyDL
#PBS -l select={num_nodes}
#PBS -l place=scatter
#PBS -l walltime={walltime}
#PBS -l filesystems=home

# Change to submission directory
cd $PBS_O_WORKDIR

# Load modules (adjust these based on your Polaris setup)
#module purge
#module load PrgEnv-gnu
#module load cray-mpich
#module load craype-x86-milan
#module load libfabric

# Set environment variables
#export MPICH_OFI_VERBOSE=1
#export MPICH_GPU_SUPPORT_ENABLED=0

echo "=============================================="
echo "Starting batch job on {num_nodes} nodes"
echo "Total simulations: {total_jobs}"
echo "=============================================="

# Create files to track job completion
JOBS_FILE="{pbs_jobs_dir}/jobs_list_{batch_id}.txt"
COMPLETED_FILE="{pbs_jobs_dir}/completed_{batch_id}.txt"
STATUS_FILE="{pbs_jobs_dir}/status_{batch_id}.txt"
touch "$COMPLETED_FILE"
touch "$STATUS_FILE"

# Function to run a single simulation
run_simulation() {{
    local job_id=$1
    local work_dir=$2
    local rotated_name=$3
    local reynolds=$4

    echo "[Job $job_id] Starting simulation: $rotated_name at Re=$reynolds" | tee -a "$STATUS_FILE"
    echo "[Job $job_id] Working directory: $work_dir" | tee -a "$STATUS_FILE"

    cd "$work_dir" || {{
        echo "[Job $job_id] ERROR: Cannot access directory $work_dir" | tee -a "$STATUS_FILE"
        echo "$job_id|FAILED|Cannot access directory" >> "$COMPLETED_FILE"
        return 1
    }}

    # Run genmap
    echo "[Job $job_id] Running genmap..." | tee -a "$STATUS_FILE"
    genmap < genmap_input.txt > genmap_${{job_id}}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "[Job $job_id] ERROR: genmap failed for $rotated_name" | tee -a "$STATUS_FILE"
        echo "$job_id|FAILED|genmap failed" >> "$COMPLETED_FILE"
        return 1
    fi

    # Initial run
    echo "[Job $job_id] Starting initial run..." | tee -a "$STATUS_FILE"
    echo $rotated_name >  SESSION.NAME
    echo `pwd`'/' >>  SESSION.NAME
    mpiexec -n 2 ./nek5000 $rotated_name > nek_initial_${{job_id}}.log 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[Job $job_id] ERROR: Initial run failed for $rotated_name" | tee -a "$STATUS_FILE"
        echo "$job_id|FAILED|Initial run failed" >> "$COMPLETED_FILE"
        return 1
    fi

    # Update par file for restart
    echo "[Job $job_id] Updating par file for restart..." | tee -a "$STATUS_FILE"
    python3 << EOF
import re

par_file = '${{rotated_name}}.par'
with open(par_file, 'r') as f:
    content = f.read()

# Update for restart
content = re.sub(r'-10000.0', '-${{reynolds}}', content)
content = re.sub(r'-10000', '-${{reynolds}}', content)
content = re.sub(r'#startFrom = rans0.f00002', 'startFrom = ${{rotated_name}}0.f00002', content)
content = re.sub(r'#timeStepper = BDF2', 'timeStepper = BDF2', content)
content = re.sub(r'#extrapolation = OIFS', 'extrapolation = OIFS', content)
content = re.sub(r'#targetCFL = 3.5.', 'targetCFL = 3.5.', content)
content = re.sub(r'numsteps = 10000', 'numsteps = 50000', content)
content = re.sub(r'writeInterval = 10000', 'writeInterval = 50000', content)

with open(par_file, 'w') as f:
    f.write(content)
EOF

    # Restart run
    echo "[Job $job_id] Starting restart run at Re=$reynolds..." | tee -a "$STATUS_FILE"
    mpiexec -n 2 ./nek5000 $rotated_name > nek_restart_${{job_id}}.log 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[Job $job_id] ERROR: Restart run failed for $rotated_name" | tee -a "$STATUS_FILE"
        echo "$job_id|FAILED|Restart run failed" >> "$COMPLETED_FILE"
        return 1
    fi

    echo "[Job $job_id] Simulation completed successfully: $rotated_name at Re=$reynolds" | tee -a "$STATUS_FILE"
    echo "$job_id|SUCCESS|Completed" >> "$COMPLETED_FILE"

    return 0
}}

# Write job information to file
cat > "$JOBS_FILE" << 'JOBSEOF'
{jobs_data}
JOBSEOF

# Process jobs in parallel using GNU parallel or xargs
echo "Processing {total_jobs} simulations in parallel..."

# Method 1: Using GNU parallel if available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job distribution"
    
    # Read jobs and run them in parallel
    cat "$JOBS_FILE" | parallel -j {jobs_per_batch} --colsep '\\|' run_simulation {{1}} {{2}} {{3}} {{4}}
else
    # Method 2: Using background processes with job control
    echo "Using background processes for parallel execution"
    
    MAX_PARALLEL={jobs_per_batch}
    current_jobs=0
    
    while IFS='|' read -r job_id work_dir rotated_name reynolds; do
        # Wait if we've reached the maximum number of parallel jobs
        while [ $current_jobs -ge $MAX_PARALLEL ]; do
            wait -n  # Wait for any background job to finish
            current_jobs=$((current_jobs - 1))
        done
        
        # Launch the simulation in the background
        (
            run_simulation "$job_id" "$work_dir" "$rotated_name" "$reynolds"
        ) &
        
        current_jobs=$((current_jobs + 1))
        
        # Brief pause to avoid overwhelming the system
        sleep 0.1
    done < "$JOBS_FILE"
    
    # Wait for all remaining jobs to complete
    wait
fi

echo "=============================================="
echo "All simulations have been processed"
echo "Generating summary..."
echo "=============================================="

# Count successes and failures
if [ -f "$COMPLETED_FILE" ]; then
    SUCCESS_COUNT=$(grep "|SUCCESS|" "$COMPLETED_FILE" 2>/dev/null | wc -l | tr -d ' ')
    FAILED_COUNT=$(grep "|FAILED|" "$COMPLETED_FILE" 2>/dev/null | wc -l | tr -d ' ')
else
    SUCCESS_COUNT=0
    FAILED_COUNT=0
fi
TOTAL_PROCESSED=$((SUCCESS_COUNT + FAILED_COUNT))

echo "Summary of results:" | tee -a "$STATUS_FILE"
echo "  Successful simulations: $SUCCESS_COUNT" | tee -a "$STATUS_FILE"
echo "  Failed simulations: $FAILED_COUNT" | tee -a "$STATUS_FILE"
echo "  Total processed: $TOTAL_PROCESSED" | tee -a "$STATUS_FILE"
echo "  Total expected: {total_jobs}" | tee -a "$STATUS_FILE"

# List failed jobs if any
if [ $FAILED_COUNT -gt 0 ]; then
    echo "" | tee -a "$STATUS_FILE"
    echo "Failed simulations:" | tee -a "$STATUS_FILE"
    grep "|FAILED|" "$COMPLETED_FILE" | while IFS='|' read -r job_id status reason; do
        echo "  Job $job_id: $reason" | tee -a "$STATUS_FILE"
    done
fi

echo "=============================================="
echo "Batch job completed"
echo "Check the following files for details:"
echo "  - $COMPLETED_FILE"
echo "  - $STATUS_FILE"
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

def create_batch_jobs(all_job_infos, num_nodes=None, walltime=None, queue="workq-route"):
    """Create PBS job scripts that distribute simulations across nodes."""
    if num_nodes is None:
        # Calculate optimal number of nodes
        total_sims = len(all_job_infos)
        # Be conservative with parallelism to avoid overloading
        sims_per_node = CORES_PER_NODE // (TASKS_PER_SIMULATION * 2)  # Factor of 2 for safety
        num_nodes = min(DEFAULT_NODES, math.ceil(total_sims / sims_per_node))
        num_nodes = max(1, num_nodes)  # At least 1 node
    
    if walltime is None:
        walltime = DEFAULT_WALLTIME
    
    # Calculate jobs per batch based on available resources
    total_cores = num_nodes * CORES_PER_NODE
    # Conservative estimate: each simulation needs 2 cores, plus overhead
    jobs_per_batch = min(len(all_job_infos), total_cores // (TASKS_PER_SIMULATION * 2))
    
    print(f"\nCreating PBS job for {len(all_job_infos)} simulations")
    print(f"  Nodes: {num_nodes}")
    print(f"  Queue: {queue}")
    print(f"  Walltime: {walltime}")
    print(f"  Max parallel simulations: {jobs_per_batch}")
    
    # Create job data string
    job_data_lines = []
    for i, job_info in enumerate(all_job_infos):
        line = f"{i}|{job_info['work_dir']}|{job_info['rotated_name']}|{job_info['reynolds']}"
        job_data_lines.append(line)
    
    jobs_data = '\n'.join(job_data_lines)
    
    # Create job script
    batch_id = "1"
    job_script = PBS_TEMPLATE.format(
        batch_id=batch_id,
        queue=queue,
        num_nodes=num_nodes,
        walltime=walltime,
        jobs_data=jobs_data,
        total_jobs=len(all_job_infos),
        jobs_per_batch=jobs_per_batch,
        pbs_jobs_dir=str(PBS_JOBS_DIR)
    )
    
    # Save job script
    job_file = PBS_JOBS_DIR / f"rans_batch_{batch_id}.pbs"
    with open(job_file, 'w') as f:
        f.write(job_script)
    
    # Make script executable
    job_file.chmod(0o755)
    
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
        print(f"{'='*60}")
        
        # Ask user about job configuration
        print(f"\nDefault configuration:")
        print(f"  - Number of nodes: {DEFAULT_NODES}")
        print(f"  - Queue: workq-route")
        print(f"  - Walltime: {DEFAULT_WALLTIME}")
        
        custom = input("\nUse custom configuration? (y/n): ").lower()
        
        num_nodes = DEFAULT_NODES
        walltime = DEFAULT_WALLTIME
        queue = "workq-route"
        
        if custom == 'y':
            # Number of nodes
            nodes_input = input(f"Number of nodes (1-{MAX_NODES}, default={DEFAULT_NODES}): ").strip()
            if nodes_input:
                try:
                    num_nodes = int(nodes_input)
                    num_nodes = max(1, min(num_nodes, MAX_NODES))
                except ValueError:
                    print(f"Invalid input, using default: {DEFAULT_NODES}")
                    num_nodes = DEFAULT_NODES
            
            # Queue selection
            print("\nAvailable queues:")
            print("  1. debug (1-8 nodes, 5min-2hr)")
            print("  2. workq-route (1-184 nodes, 45min-24hr)")
            print("  3. preemptable (1-10 nodes, 5min-72hr, can be killed)")
            queue_choice = input("Select queue (1-3, default=2): ").strip()
            
            if queue_choice == '1':
                queue = "debug"
                num_nodes = min(num_nodes, 8)
                max_walltime = "2:00:00"
            elif queue_choice == '3':
                queue = "preemptable"
                num_nodes = min(num_nodes, 10)
                max_walltime = "72:00:00"
            else:
                queue = "workq-route"
                max_walltime = "24:00:00"
            
            # Walltime
            walltime_input = input(f"Walltime (HH:MM:SS, default={DEFAULT_WALLTIME}): ").strip()
            if walltime_input:
                walltime = walltime_input

        # Create batch job
        batch_files = create_batch_jobs(all_job_infos, num_nodes=num_nodes, 
                                       walltime=walltime, queue=queue)
        
        print(f"\nBatch job configuration:")
        print(f"  - {len(all_job_infos)} simulations")
        print(f"  - {num_nodes} nodes")
        print(f"  - Queue: {queue}")
        print(f"  - Walltime: {walltime}")

        # Ask user about job submission
        response = input("\nSubmit batch job to PBS? (y/n): ").lower()
        if response == 'y':
            print("\nSubmitting job...")
            job_ids = submit_jobs(batch_files)

            print(f"\n{'='*60}")
            print(f"Submitted batch job successfully!")
            print("Monitor progress with: qstat -u $USER")
            print(f"Check outputs in: {PBS_JOBS_DIR}/")
            print(f"  - Output: rans_batch_1.o")
            print(f"  - Errors: rans_batch_1.e")
            print(f"  - Status: status_1.txt")
            print(f"  - Results: completed_1.txt")
            print(f"{'='*60}")
        else:
            print("\nBatch job script created but not submitted.")
            print(f"To submit manually, run: qsub {batch_files[0]}")
            print(f"Batch script is in: {PBS_JOBS_DIR}")
    else:
        print("\nNo simulations to run.")

if __name__ == "__main__":
    main()
