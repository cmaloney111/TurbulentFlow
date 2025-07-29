#!/usr/bin/env python3
"""
Optimized Parallelized RANS Simulation Automation Script for SLURM
Separates initial runs (Re=10k) from restart runs to avoid redundant computations
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
from collections import defaultdict

# Configuration - List of airfoils to process
AIRFOILS_TO_PROCESS = [
    "aquila",
    "clark-y",
    "dae51",
    "df101",
    "df102",
    "e193",
    "fx60-100",
    "j5012",
    "mb253515",
]

# Directory paths
SCRIPT_DIR = Path(__file__).parent.absolute()
GENAIRFOILS_DIR = SCRIPT_DIR / "GenAirfoils"
AIRFOIL_DB_DIR = GENAIRFOILS_DIR / "airfoil_database"
RE2_FILES_DIR = GENAIRFOILS_DIR / "re2_files"
RANS_BASE_DIR = SCRIPT_DIR / "rans_base"
INITIAL_RANS_RUNS_DIR = SCRIPT_DIR / "initial_rans_runs"
RESTART_RANS_RUNS_DIR = SCRIPT_DIR / "restart_rans_runs"
CSV_FILE = SCRIPT_DIR / "training_data_stec8.csv"
SLURM_JOBS_DIR = SCRIPT_DIR / "slurm_jobs"

# SLURM configuration
CORES_PER_NODE = 128
TASKS_PER_SIMULATION = 2
DEFAULT_NODES = 2
DEFAULT_WALLTIME = "00:45:00"
INITIAL_REYNOLDS = 10000  # Reynolds number for initial runs

SLURM_INITIAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name="rans_initial_{batch_id}"
#SBATCH --output="{slurm_jobs_dir}/rans_initial_{batch_id}.%j.out"
#SBATCH --error="{slurm_jobs_dir}/rans_initial_{batch_id}.%j.err"
#SBATCH --partition={partition}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={cores_per_node}
#SBATCH --account=slo102
#SBATCH --export=ALL
#SBATCH --time={walltime}

module purge
module load slurm
module load cpu/0.15.4
module load intel/19.1.1.217
module load openmpi/3.1.6

echo "=============================================="
echo "Starting INITIAL runs on {num_nodes} nodes"
echo "Total simulations: {total_jobs}"
echo "=============================================="

# Create files to track job completion
JOBS_FILE="{slurm_jobs_dir}/initial_jobs_list_{batch_id}.txt"
COMPLETED_FILE="{slurm_jobs_dir}/initial_completed_{batch_id}.txt"
STATUS_FILE="{slurm_jobs_dir}/initial_status_{batch_id}.txt"
touch "$COMPLETED_FILE"
touch "$STATUS_FILE"

# Write job information to file
cat > "$JOBS_FILE" << 'JOBSEOF'
{jobs_data}
JOBSEOF

# Get total number of nodes
NNODES=$SLURM_JOB_NUM_NODES

# Settings for each simulation
NRANKS_PER_NODE={tasks_per_simulation}
CORES_PER_NODE={cores_per_node}
SIMS_PER_NODE=$((CORES_PER_NODE / NRANKS_PER_NODE))
NDEPTH=$((CORES_PER_NODE / SIMS_PER_NODE))

echo "NUM_OF_NODES= ${{NNODES}}"
echo "SIMULATIONS_PER_NODE= ${{SIMS_PER_NODE}}"
echo "RANKS_PER_SIMULATION= ${{NRANKS_PER_NODE}}"
echo "CORES_PER_SIMULATION= ${{NDEPTH}}"

# Read all jobs into an array
mapfile -t JOBS_ARRAY < "$JOBS_FILE"
TOTAL_JOBS=${{#JOBS_ARRAY[@]}}

# Get list of nodes
NODES_ARRAY=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Distribute jobs across all nodes
JOB_INDEX=0
while [ $JOB_INDEX -lt $TOTAL_JOBS ]; do
    for NODE_INDEX in $(seq 0 $((NNODES - 1))); do
        NODE="${{NODES_ARRAY[$NODE_INDEX]}}"
        
        for sim_on_node in $(seq 0 $((SIMS_PER_NODE - 1))); do
            if [ $JOB_INDEX -ge $TOTAL_JOBS ]; then
                break 2
            fi
            
            # Parse job information
            IFS='|' read -r job_id work_dir rotated_name <<< "${{JOBS_ARRAY[$JOB_INDEX]}}"
            
            echo "[Node $((NODE_INDEX + 1)), Sim $sim_on_node] Starting initial job $job_id: $rotated_name on $NODE"
            
            # Launch the initial simulation
            (
                cd "$work_dir" || {{
                    echo "[Job $job_id] ERROR: Cannot access directory $work_dir" | tee -a "$STATUS_FILE"
                    echo "$job_id|FAILED|Cannot access directory" >> "$COMPLETED_FILE"
                    exit 1
                }}
                
                # Run genmap
                echo "[Job $job_id] Running genmap..." >> "$STATUS_FILE"
                genmap < genmap_input.txt > genmap_${{job_id}}.log 2>&1
                
                if [ $? -ne 0 ]; then
                    echo "[Job $job_id] ERROR: genmap failed for $rotated_name" | tee -a "$STATUS_FILE"
                    echo "$job_id|FAILED|genmap failed" >> "$COMPLETED_FILE"
                    exit 1
                fi
                
                # Initial run at Re=10000
                echo "[Job $job_id] Starting initial run at Re=10000..." >> "$STATUS_FILE"
                echo $rotated_name > SESSION.NAME
                echo `pwd`'/' >> SESSION.NAME
                
                # Calculate CPU binding offset for this simulation on the node
                CPU_OFFSET=$((sim_on_node * NDEPTH))
                
                # Run with specific node and CPU binding
                srun --exclusive --nodelist=$NODE --ntasks=${{NRANKS_PER_NODE}} \
                     --cpus-per-task=$((NDEPTH / NRANKS_PER_NODE)) \
                     --cpu-bind=map_cpu:$((CPU_OFFSET)),$((CPU_OFFSET + NDEPTH / NRANKS_PER_NODE)) \
                     ./nek5000 $rotated_name > nek_initial_${{job_id}}.log 2>&1
                
                if [ $? -ne 0 ]; then
                    echo "[Job $job_id] ERROR: Initial run failed for $rotated_name" | tee -a "$STATUS_FILE"
                    echo "$job_id|FAILED|Initial run failed" >> "$COMPLETED_FILE"
                    exit 1
                fi
                
                echo "[Job $job_id] Initial simulation completed successfully: $rotated_name" | tee -a "$STATUS_FILE"
                echo "$job_id|SUCCESS|Completed" >> "$COMPLETED_FILE"
            ) &
            
            JOB_INDEX=$((JOB_INDEX + 1))
            sleep 0.05
        done
    done
    sleep 0.1
done

# Wait for all simulations to complete
echo "Waiting for all initial simulations to complete..."
wait

echo "=============================================="
echo "All initial simulations completed"
echo "=============================================="

# Count successes and failures
if [ -f "$COMPLETED_FILE" ]; then
    SUCCESS_COUNT=$(grep "|SUCCESS|" "$COMPLETED_FILE" 2>/dev/null | wc -l | tr -d ' ')
    FAILED_COUNT=$(grep "|FAILED|" "$COMPLETED_FILE" 2>/dev/null | wc -l | tr -d ' ')
else
    SUCCESS_COUNT=0
    FAILED_COUNT=0
fi

echo "Summary of results:" | tee -a "$STATUS_FILE"
echo "  Successful simulations: $SUCCESS_COUNT" | tee -a "$STATUS_FILE"
echo "  Failed simulations: $FAILED_COUNT" | tee -a "$STATUS_FILE"
echo "  Total expected: {total_jobs}" | tee -a "$STATUS_FILE"
"""

SLURM_RESTART_TEMPLATE = """#!/bin/bash
#SBATCH --job-name="rans_restart_{batch_id}"
#SBATCH --output="{slurm_jobs_dir}/rans_restart_{batch_id}.%j.out"
#SBATCH --error="{slurm_jobs_dir}/rans_restart_{batch_id}.%j.err"
#SBATCH --partition={partition}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={cores_per_node}
#SBATCH --account=slo102
#SBATCH --export=ALL
#SBATCH --time={walltime}

module purge
module load slurm
module load cpu/0.15.4
module load intel/19.1.1.217
module load openmpi/3.1.6

echo "=============================================="
echo "Starting RESTART runs on {num_nodes} nodes"
echo "Total simulations: {total_jobs}"
echo "=============================================="

# Create files to track job completion
JOBS_FILE="{slurm_jobs_dir}/restart_jobs_list_{batch_id}.txt"
COMPLETED_FILE="{slurm_jobs_dir}/restart_completed_{batch_id}.txt"
STATUS_FILE="{slurm_jobs_dir}/restart_status_{batch_id}.txt"
touch "$COMPLETED_FILE"
touch "$STATUS_FILE"

# Write job information to file
cat > "$JOBS_FILE" << 'JOBSEOF'
{jobs_data}
JOBSEOF

# Get total number of nodes
NNODES=$SLURM_JOB_NUM_NODES

# Settings for each simulation
NRANKS_PER_NODE={tasks_per_simulation}
CORES_PER_NODE={cores_per_node}
SIMS_PER_NODE=$((CORES_PER_NODE / NRANKS_PER_NODE))
NDEPTH=$((CORES_PER_NODE / SIMS_PER_NODE))

echo "NUM_OF_NODES= ${{NNODES}}"
echo "SIMULATIONS_PER_NODE= ${{SIMS_PER_NODE}}"
echo "RANKS_PER_SIMULATION= ${{NRANKS_PER_NODE}}"
echo "CORES_PER_SIMULATION= ${{NDEPTH}}"

# Read all jobs into an array
mapfile -t JOBS_ARRAY < "$JOBS_FILE"
TOTAL_JOBS=${{#JOBS_ARRAY[@]}}

# Get list of nodes
NODES_ARRAY=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Distribute jobs across all nodes
JOB_INDEX=0
while [ $JOB_INDEX -lt $TOTAL_JOBS ]; do
    for NODE_INDEX in $(seq 0 $((NNODES - 1))); do
        NODE="${{NODES_ARRAY[$NODE_INDEX]}}"
        
        for sim_on_node in $(seq 0 $((SIMS_PER_NODE - 1))); do
            if [ $JOB_INDEX -ge $TOTAL_JOBS ]; then
                break 2
            fi
            
            # Parse job information
            IFS='|' read -r job_id work_dir rotated_name reynolds initial_dir <<< "${{JOBS_ARRAY[$JOB_INDEX]}}"
            
            echo "[Node $((NODE_INDEX + 1)), Sim $sim_on_node] Starting restart job $job_id: $rotated_name at Re=$reynolds on $NODE"
            
            # Launch the restart simulation
            (
                cd "$work_dir" || {{
                    echo "[Job $job_id] ERROR: Cannot access directory $work_dir" | tee -a "$STATUS_FILE"
                    echo "$job_id|FAILED|Cannot access directory" >> "$COMPLETED_FILE"
                    exit 1
                }}
                
                # Copy files from initial run
                echo "[Job $job_id] Copying files from initial run..." >> "$STATUS_FILE"
                cp -r "$initial_dir"/* . 2>/dev/null || true
                
                # Update par file for restart
                echo "[Job $job_id] Updating par file for restart at Re=$reynolds..." >> "$STATUS_FILE"
                python3 << EOF
import re

par_file = '${{rotated_name}}.par'
with open(par_file, 'r') as f:
    content = f.read()

# Update for restart
content = re.sub(r'-10000\.0', '-${{reynolds}}', content)
content = re.sub(r'-10000', '-${{reynolds}}', content)
content = re.sub(r'dt = 5.0e-7', 'dt = 1.0e-5', content)
content = re.sub(r'#startFrom = rans0.f00001', 'startFrom = ${{rotated_name}}0.f00002', content)
content = re.sub(r'#timeStepper = BDF2', 'timeStepper = BDF2', content)
content = re.sub(r'#extrapolation = OIFS', 'extrapolation = OIFS', content)
content = re.sub(r'#targetCFL = 3.5.', 'targetCFL = 3.5.', content)
content = re.sub(r'numsteps = 10', 'numsteps = 100', content)
content = re.sub(r'writeInterval = 10', 'writeInterval = 100', content)

with open(par_file, 'w') as f:
    f.write(content)
EOF
                
                # Restart run
                echo "[Job $job_id] Starting restart run at Re=$reynolds..." >> "$STATUS_FILE"
                echo $rotated_name > SESSION.NAME
                echo `pwd`'/' >> SESSION.NAME
                
                # Calculate CPU binding offset for this simulation on the node
                CPU_OFFSET=$((sim_on_node * NDEPTH))
                
                # Run with specific node and CPU binding
                srun --exclusive --nodelist=$NODE --ntasks=${{NRANKS_PER_NODE}} \
                     --cpus-per-task=$((NDEPTH / NRANKS_PER_NODE)) \
                     --cpu-bind=map_cpu:$((CPU_OFFSET)),$((CPU_OFFSET + NDEPTH / NRANKS_PER_NODE)) \
                     ./nek5000 $rotated_name > nek_restart_${{job_id}}.log 2>&1
                
                if [ $? -ne 0 ]; then
                    echo "[Job $job_id] ERROR: Restart run failed for $rotated_name" | tee -a "$STATUS_FILE"
                    echo "$job_id|FAILED|Restart run failed" >> "$COMPLETED_FILE"
                    exit 1
                fi
                
                echo "[Job $job_id] Restart simulation completed successfully: $rotated_name at Re=$reynolds" | tee -a "$STATUS_FILE"
                echo "$job_id|SUCCESS|Completed" >> "$COMPLETED_FILE"
            ) &
            
            JOB_INDEX=$((JOB_INDEX + 1))
            sleep 0.05
        done
    done
    sleep 0.1
done

# Wait for all simulations to complete
echo "Waiting for all restart simulations to complete..."
wait

echo "=============================================="
echo "All restart simulations completed"
echo "=============================================="

# Count successes and failures
if [ -f "$COMPLETED_FILE" ]; then
    SUCCESS_COUNT=$(grep "|SUCCESS|" "$COMPLETED_FILE" 2>/dev/null | wc -l | tr -d ' ')
    FAILED_COUNT=$(grep "|FAILED|" "$COMPLETED_FILE" 2>/dev/null | wc -l | tr -d ' ')
else
    SUCCESS_COUNT=0
    FAILED_COUNT=0
fi

echo "Summary of results:" | tee -a "$STATUS_FILE"
echo "  Successful simulations: $SUCCESS_COUNT" | tee -a "$STATUS_FILE"
echo "  Failed simulations: $FAILED_COUNT" | tee -a "$STATUS_FILE"
echo "  Total expected: {total_jobs}" | tee -a "$STATUS_FILE"
"""

def run_command(cmd, cwd=None, input_text=None, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, cwd=cwd, universal_newlines=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              input=input_text)
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
    all_angles = set()

    for reynolds in airfoil_data['reynolds_number'].unique():
        reynolds_data = airfoil_data[airfoil_data['reynolds_number'] == reynolds]
        angles_rad = reynolds_data['angle_of_attack'].unique()
        angles_deg = [int(np.floor(angle)) for angle in angles_rad]
        reynolds_to_angles[reynolds] = sorted(angles_deg)
        all_angles.update(angles_deg)

    return reynolds_to_angles, sorted(all_angles)

def rotate_airfoil(airfoil_name, angle):
    """Rotate an airfoil using rotate_dat.py."""
    if angle == 0:
        return airfoil_name

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
    re2_file = RE2_FILES_DIR / f"{airfoil_name}.re2"

    if not re2_file.exists():
        cmd = [
            "python",
            str(GENAIRFOILS_DIR / "test_gen.py"),
            "--airfoil", airfoil_name
        ]
        run_command(cmd, cwd=GENAIRFOILS_DIR)

def create_initial_directory(airfoil_name, angle):
    """Create directory structure for initial RANS run."""
    angle_dir = str(angle).replace("-", "neg")
    initial_dir = INITIAL_RANS_RUNS_DIR / airfoil_name / angle_dir
    initial_dir.mkdir(parents=True, exist_ok=True)
    return initial_dir

def create_restart_directory(airfoil_name, angle, reynolds):
    """Create directory structure for restart RANS run."""
    angle_dir = str(angle).replace("-", "neg")
    restart_dir = RESTART_RANS_RUNS_DIR / airfoil_name / f"Re{int(reynolds)}" / angle_dir
    restart_dir.mkdir(parents=True, exist_ok=True)
    return restart_dir

def prepare_initial_simulation_files(initial_dir, airfoil_name, angle):
    """Prepare files for initial simulation at Re=10000."""
    # Copy all files from rans_base
    for file in RANS_BASE_DIR.iterdir():
        if file.is_file():
            dest = initial_dir / file.name
            if file.name == "nek5000":
                # Create symlink for executable
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
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
    shutil.copy2(re2_source, initial_dir)

    # Create genmap input file
    genmap_input_file = initial_dir / "genmap_input.txt"
    with open(genmap_input_file, 'w') as f:
        f.write(f"{rotated_name}\n0.05\n")

    # Prepare initial par file (with Re=10000)
    old_par = initial_dir / "rans.par"
    new_par = initial_dir / f"{rotated_name}.par"
    shutil.copy2(old_par, new_par)
    old_par.unlink()

    return rotated_name

def create_batch_jobs(initial_job_infos, restart_job_infos, num_nodes=None, 
                     walltime=None, partition="shared"):
    """Create SLURM job scripts for initial and restart runs."""
    if num_nodes is None:
        num_nodes = DEFAULT_NODES
    
    if walltime is None:
        walltime = DEFAULT_WALLTIME
    
    sims_per_node = CORES_PER_NODE // TASKS_PER_SIMULATION
    max_parallel_sims = num_nodes * sims_per_node
    
    job_files = []
    
    # Create initial runs job if needed
    if initial_job_infos:
        print(f"\nCreating SLURM job for {len(initial_job_infos)} initial simulations")
        print(f"  Nodes: {num_nodes}")
        print(f"  Partition: {partition}")
        print(f"  Walltime: {walltime}")
        print(f"  Max parallel simulations: {max_parallel_sims}")
        
        # Create job data string for initial runs
        job_data_lines = []
        for i, job_info in enumerate(initial_job_infos):
            line = f"{i}|{job_info['work_dir']}|{job_info['rotated_name']}"
            job_data_lines.append(line)
        
        jobs_data = '\n'.join(job_data_lines)
        
        # Create initial job script
        batch_id = "1"
        job_script = SLURM_INITIAL_TEMPLATE.format(
            batch_id=batch_id,
            partition=partition,
            num_nodes=num_nodes,
            walltime=walltime,
            jobs_data=jobs_data,
            total_jobs=len(initial_job_infos),
            slurm_jobs_dir=str(SLURM_JOBS_DIR),
            tasks_per_simulation=TASKS_PER_SIMULATION,
            cores_per_node=CORES_PER_NODE
        )
        
        # Save job script
        job_file = SLURM_JOBS_DIR / f"rans_initial_{batch_id}.sbatch"
        with open(job_file, 'w') as f:
            f.write(job_script)
        
        job_file.chmod(0o755)
        job_files.append(('initial', job_file))
    
    # Create restart runs job if needed
    if restart_job_infos:
        print(f"\nCreating SLURM job for {len(restart_job_infos)} restart simulations")
        
        # Create job data string for restart runs
        job_data_lines = []
        for i, job_info in enumerate(restart_job_infos):
            line = f"{i}|{job_info['work_dir']}|{job_info['rotated_name']}|{job_info['reynolds']}|{job_info['initial_dir']}"
            job_data_lines.append(line)
        
        jobs_data = '\n'.join(job_data_lines)
        
        # Create restart job script
        batch_id = "1"
        job_script = SLURM_RESTART_TEMPLATE.format(
            batch_id=batch_id,
            partition=partition,
            num_nodes=num_nodes,
            walltime=walltime,
            jobs_data=jobs_data,
            total_jobs=len(restart_job_infos),
            slurm_jobs_dir=str(SLURM_JOBS_DIR),
            tasks_per_simulation=TASKS_PER_SIMULATION,
            cores_per_node=CORES_PER_NODE
        )
        
        # Save job script
        job_file = SLURM_JOBS_DIR / f"rans_restart_{batch_id}.sbatch"
        with open(job_file, 'w') as f:
            f.write(job_script)
        
        job_file.chmod(0o755)
        job_files.append(('restart', job_file))
    
    return job_files

def prepare_all_simulations(airfoil_name, df):
    """Prepare initial and restart simulations for an airfoil."""
    print(f"\n{'='*60}")
    print(f"Preparing simulations for: {airfoil_name}")
    print(f"{'='*60}")

    reynolds_to_angles, all_angles = get_angles_and_reynolds(airfoil_name, df)
    initial_job_infos = []
    restart_job_infos = []
    
    # Track initial directories for each angle
    angle_to_initial_dir = {}

    # First: Prepare all rotated airfoils and RE2 files
    print("\nPhase 1: Rotating airfoils and converting to RE2...")
    for angle in all_angles:
        print(f"  Preparing {airfoil_name} at {angle}°")
        rotated_name = rotate_airfoil(airfoil_name, angle)
        convert_to_re2(rotated_name)

    # Second: Prepare initial runs (one per angle)
    print("\nPhase 2: Preparing initial runs at Re=10000...")
    for angle in all_angles:
        initial_dir = create_initial_directory(airfoil_name, angle)
        rotated_name = prepare_initial_simulation_files(initial_dir, airfoil_name, angle)
        
        angle_to_initial_dir[angle] = initial_dir
        
        initial_job_info = {
            'airfoil': airfoil_name,
            'angle': angle,
            'rotated_name': rotated_name,
            'work_dir': str(initial_dir)
        }
        
        initial_job_infos.append(initial_job_info)
        print(f"  Prepared initial simulation: {rotated_name}")

    # Third: Prepare restart runs (for all reynolds numbers)
    print("\nPhase 3: Preparing restart runs...")
    for reynolds, angles in reynolds_to_angles.items():
        # Skip if this is the initial Reynolds number
        if reynolds == INITIAL_REYNOLDS:
            continue
            
        for angle in angles:
            restart_dir = create_restart_directory(airfoil_name, angle, reynolds)
            
            if angle == 0:
                rotated_name = airfoil_name
            else:
                rotated_name = f"{airfoil_name}_rot{angle}"
            
            restart_job_info = {
                'airfoil': airfoil_name,
                'angle': angle,
                'reynolds': reynolds,
                'rotated_name': rotated_name,
                'work_dir': str(restart_dir),
                'initial_dir': str(angle_to_initial_dir[angle])
            }
            
            restart_job_infos.append(restart_job_info)
            print(f"  Prepared restart simulation: {rotated_name} at Re={reynolds}")

    return initial_job_infos, restart_job_infos

def submit_jobs(job_files, submit_restart_after_initial=True):
    """Submit SLURM jobs with optional dependency."""
    job_ids = {}
    
    for job_type, job_file in job_files:
        cmd = ["sbatch"]
        
        # If this is a restart job and we have an initial job, add dependency
        if job_type == 'restart' and 'initial' in job_ids and submit_restart_after_initial:
            cmd.extend(["--dependency", f"afterok:{job_ids['initial']}"])
        
        cmd.append(str(job_file))
        
        result = run_command(cmd, check=False)
        
        if result and result.returncode == 0:
            # Extract job ID from output
            output = result.stdout.strip()
            job_id = output.split()[-1]
            job_ids[job_type] = job_id
            print(f"Submitted {job_type} job {job_file.name} with ID: {job_id}")
            if job_type == 'restart' and 'initial' in job_ids and submit_restart_after_initial:
                print(f"  (Will start after initial job {job_ids['initial']} completes)")
        else:
            print(f"Failed to submit {job_type} job {job_file.name}")
            if result:
                print(f"Error: {result.stderr}")
    
    return job_ids

def main():
    """Main function to orchestrate the workflow."""
    print("Optimized Parallelized RANS Simulation Automation Script for SLURM")
    print("=================================================================")

    # Create necessary directories
    INITIAL_RANS_RUNS_DIR.mkdir(exist_ok=True)
    RESTART_RANS_RUNS_DIR.mkdir(exist_ok=True)
    SLURM_JOBS_DIR.mkdir(exist_ok=True)

    # Load CSV data
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    all_initial_job_infos = []
    all_restart_job_infos = []

    # Prepare all simulations
    for airfoil_name in AIRFOILS_TO_PROCESS:
        if check_airfoil_exists(airfoil_name, df):
            initial_jobs, restart_jobs = prepare_all_simulations(airfoil_name, df)
            all_initial_job_infos.extend(initial_jobs)
            all_restart_job_infos.extend(restart_jobs)
        else:
            print(f"\nSkipping {airfoil_name}: Not found in both CSV and database")

    if all_initial_job_infos or all_restart_job_infos:
        print(f"\n{'='*60}")
        print(f"Prepared {len(all_initial_job_infos)} initial simulations")
        print(f"Prepared {len(all_restart_job_infos)} restart simulations")
        print(f"Total: {len(all_initial_job_infos) + len(all_restart_job_infos)} simulations")
        print(f"{'='*60}")
        
        # Ask user about job configuration
        print(f"\nDefault configuration:")
        print(f"  - Number of nodes: {DEFAULT_NODES}")
        print(f"  - Partition: shared")
        print(f"  - Walltime: {DEFAULT_WALLTIME}")
        
        custom = input("\nUse custom configuration? (y/n): ").lower()
        
        num_nodes = DEFAULT_NODES
        walltime = DEFAULT_WALLTIME
        partition = "shared"
        
        if custom == 'y':
            # Number of nodes
            nodes_input = input(f"Number of nodes (default={DEFAULT_NODES}): ").strip()
            if nodes_input:
                try:
                    num_nodes = int(nodes_input)
                    num_nodes = max(1, num_nodes)
                except ValueError:
                    print(f"Invalid input, using default: {DEFAULT_NODES}")
                    num_nodes = DEFAULT_NODES
            
            # Partition selection
            print("\nAvailable partitions:")
            print("  1. shared (default)")
            print("  2. compute")
            print("  3. debug")
            partition_choice = input("Select partition (1-3, default=1): ").strip()
            
            if partition_choice == '2':
                partition = "compute"
            elif partition_choice == '3':
                partition = "debug"
            else:
                partition = "shared"
            
            # Walltime
            walltime_input = input(f"Walltime (HH:MM:SS, default={DEFAULT_WALLTIME}): ").strip()
            if walltime_input:
                walltime = walltime_input

        # Create batch jobs
        batch_files = create_batch_jobs(all_initial_job_infos, all_restart_job_infos,
                                       num_nodes=num_nodes, walltime=walltime, partition=partition)
        
        print(f"\nBatch job configuration:")
        if all_initial_job_infos:
            print(f"  - {len(all_initial_job_infos)} initial simulations")
        if all_restart_job_infos:
            print(f"  - {len(all_restart_job_infos)} restart simulations")
        print(f"  - {num_nodes} nodes per job")
        print(f"  - Partition: {partition}")
        print(f"  - Walltime: {walltime}")

        # Ask user about job submission
        response = input("\nSubmit batch jobs to SLURM? (y/n): ").lower()
        if response == 'y':
            # Ask about dependency
            submit_with_dependency = True
            if all_initial_job_infos and all_restart_job_infos:
                dep_response = input("Submit restart job with dependency on initial job? (y/n, default=y): ").lower()
                submit_with_dependency = dep_response != 'n'
            
            print("\nSubmitting jobs...")
            job_ids = submit_jobs(batch_files, submit_restart_after_initial=submit_with_dependency)

            print(f"\n{'='*60}")
            print(f"Submitted batch jobs successfully!")
            print("Monitor progress with: squeue -u $USER")
            print(f"Check outputs in: {SLURM_JOBS_DIR}/")
            
            if 'initial' in job_ids:
                print(f"\nInitial runs:")
                print(f"  - Output: rans_initial_1.*.out")
                print(f"  - Errors: rans_initial_1.*.err")
                print(f"  - Status: initial_status_1.txt")
                print(f"  - Results: initial_completed_1.txt")
            
            if 'restart' in job_ids:
                print(f"\nRestart runs:")
                print(f"  - Output: rans_restart_1.*.out")
                print(f"  - Errors: rans_restart_1.*.err")
                print(f"  - Status: restart_status_1.txt")
                print(f"  - Results: restart_completed_1.txt")
            
            print(f"{'='*60}")
        else:
            print("\nBatch job scripts created but not submitted.")
            print(f"To submit manually:")
            for job_type, job_file in batch_files:
                print(f"  sbatch {job_file}")
            print(f"Scripts are in: {SLURM_JOBS_DIR}")
    else:
        print("\nNo simulations to run.")

if __name__ == "__main__":
    main()
