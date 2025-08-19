#!/usr/bin/env python3
"""
Optimized Parallelized RANS Simulation Automation Script for Polaris
Modified to rotate velocity instead of airfoil mesh
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
AIRFOILS_TO_PROCESS = ['s2055', 's3016', 's4062', 's4180', 's4233', 'sd2030', 'sd2083', 'sd5060', 'sd6060', 'sd6080', 'spica', 'rg15']

# Directory paths
SCRIPT_DIR = Path(__file__).parent.absolute()
GENAIRFOILS_DIR = SCRIPT_DIR / "GenAirfoils"
AIRFOIL_DB_DIR = GENAIRFOILS_DIR / "airfoil_database"
RE2_FILES_DIR = GENAIRFOILS_DIR / "re2_files"
RANS_BASE_DIR = SCRIPT_DIR / "rans_base"
INITIAL_RANS_RUNS_DIR = SCRIPT_DIR / "initial_rans_runs"
RESTART_RANS_RUNS_DIR = SCRIPT_DIR / "restart_rans_runs"
CSV_FILE = SCRIPT_DIR / "training_data_stec8.csv"
PBS_JOBS_DIR = SCRIPT_DIR / "pbs_jobs"

CORES_PER_NODE = 128
TASKS_PER_SIMULATION = 16
MAX_NODES = 184  # Maximum for workq-route
DEFAULT_NODES = 176  # Default number of nodes to request
DEFAULT_WALLTIME = "24:00:00"  # Default walltime
INITIAL_REYNOLDS = 10000  # Reynolds number for initial runs

PBS_INITIAL_TEMPLATE = """#!/bin/bash
#PBS -N rans_initial_{batch_id}
#PBS -o {pbs_jobs_dir}/rans_initial_{batch_id}.o
#PBS -e {pbs_jobs_dir}/rans_initial_{batch_id}.e
#PBS -q {queue}
#PBS -A insitu
#PBS -l select={num_nodes}
#PBS -l place=scatter
#PBS -l walltime=15:00:00
#PBS -l filesystems=home

# Change to submission directory
cd $PBS_O_WORKDIR
./save_memory.sh &
SAVE_MEMORY_PID=$!

echo "=============================================="
echo "Starting INITIAL runs on {num_nodes} nodes"
echo "Total simulations: {total_jobs}"
echo "=============================================="

# Create files to track job completion
JOBS_FILE="{pbs_jobs_dir}/initial_jobs_list_{batch_id}.txt"
COMPLETED_FILE="{pbs_jobs_dir}/initial_completed_{batch_id}.txt"
STATUS_FILE="{pbs_jobs_dir}/initial_status_{batch_id}.txt"
touch "$COMPLETED_FILE"
touch "$STATUS_FILE"

# Write job information to file
cat > "$JOBS_FILE" << 'JOBSEOF'
{jobs_data}
JOBSEOF

# Get total number of nodes
NNODES=`wc -l < $PBS_NODEFILE`

# Settings for each simulation
NRANKS_PER_NODE={tasks_per_simulation}
CORES_PER_NODE={cores_per_node}
SIMS_PER_NODE=$((CORES_PER_NODE / NRANKS_PER_NODE))
NDEPTH={tasks_per_simulation}

echo "NUM_OF_NODES= ${{NNODES}}"
echo "SIMULATIONS_PER_NODE= ${{SIMS_PER_NODE}}"
echo "RANKS_PER_SIMULATION= ${{NRANKS_PER_NODE}}"
echo "CORES_PER_SIMULATION= ${{NDEPTH}}"

# Read all jobs into an array
mapfile -t JOBS_ARRAY < "$JOBS_FILE"
TOTAL_JOBS=${{#JOBS_ARRAY[@]}}

# Read nodes into an array
mapfile -t NODES_ARRAY < $PBS_NODEFILE

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
            IFS='|' read -r job_id work_dir airfoil_name <<< "${{JOBS_ARRAY[$JOB_INDEX]}}"
            
            echo "[Node $((NODE_INDEX + 1)), Sim $sim_on_node] Starting initial job $job_id: $airfoil_name on $NODE"
            
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
                    echo "[Job $job_id] ERROR: genmap failed for $airfoil_name" | tee -a "$STATUS_FILE"
                    echo "$job_id|FAILED|genmap failed" >> "$COMPLETED_FILE"
                    exit 1
                fi
               
                # Clean up directory before initial run
                find . ! \\( -name "*.ma2" -o -name "*.f00001" -o -name "*.re2" -o -name "*.par" -o -name "nek5000" \\) -delete 
                
                # Initial run at Re=10000
                echo "[Job ${{job_id}}] Starting initial run at Re=10000..." >> "${{STATUS_FILE}}"
                echo ${{airfoil_name}} > SESSION.NAME
                echo $(pwd)'/' >> SESSION.NAME

                # Create a hostfile for this specific simulation
                echo "${{NODE}}" > hostfile_job_${{job_id}}

                MPI_ARG="-n ${{NRANKS_PER_NODE}} --ppn ${{NRANKS_PER_NODE}} --depth=${{NDEPTH}} --cpu-bind depth"

                mpiexec ${{MPI_ARG}} --hostfile hostfile_job_${{job_id}} ./nek5000 $airfoil_name > /dev/null 2>&1
                
                if [ ${{?}} -ne 0 ]; then
                    echo "[Job ${{job_id}}] ERROR: Initial run failed for ${{airfoil_name}}" | tee -a "${{STATUS_FILE}}"
                    echo "${{job_id}}|FAILED|Initial run failed" >> "${{COMPLETED_FILE}}"
                    rm -f hostfile_job_${{job_id}}
                    exit 1
                fi

                echo "[Job $job_id] Initial simulation completed successfully: $airfoil_name" | tee -a "$STATUS_FILE"
                echo "$job_id|SUCCESS|Completed" >> "$COMPLETED_FILE"
                
                # Clean up
                rm -f hostfile_job_${{job_id}} ${{airfoil_name}}0.f00001 drag.txt nek_initial.log
                echo "Step     100000,\n" > nek_initial.log

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

# Stop save_memory.sh
kill $SAVE_MEMORY_PID 2>/dev/null || true

echo "=============================================="
echo "All initial simulations completed"
echo "=============================================="

exit 0
"""

PBS_RESTART_TEMPLATE = """#!/bin/bash
#PBS -N rans_restart_{batch_id}
#PBS -o {pbs_jobs_dir}/rans_restart_{batch_id}.o
#PBS -e {pbs_jobs_dir}/rans_restart_{batch_id}.e
#PBS -q {queue}
#PBS -A insitu
#PBS -l select={num_nodes}
#PBS -l place=scatter
#PBS -l walltime={walltime}
#PBS -l filesystems=home

# Change to submission directory
cd $PBS_O_WORKDIR

# Start save_memory.sh in background
./save_memory.sh &
SAVE_MEMORY_PID=$!

echo "=============================================="
echo "Starting RESTART runs on {num_nodes} nodes"
echo "Total simulations: {total_jobs}"
echo "=============================================="

# Create files to track job completion
JOBS_FILE="{pbs_jobs_dir}/restart_jobs_list_{batch_id}.txt"
COMPLETED_FILE="{pbs_jobs_dir}/restart_completed_{batch_id}.txt"
STATUS_FILE="{pbs_jobs_dir}/restart_status_{batch_id}.txt"
touch "$COMPLETED_FILE"
touch "$STATUS_FILE"

# Write job information to file
cat > "$JOBS_FILE" << 'JOBSEOF'
{jobs_data}
JOBSEOF

# Get total number of nodes
NNODES=`wc -l < $PBS_NODEFILE`

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

# Read nodes into an array
mapfile -t NODES_ARRAY < $PBS_NODEFILE

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
            IFS='|' read -r job_id work_dir airfoil_name reynolds initial_dir <<< "${{JOBS_ARRAY[$JOB_INDEX]}}"
            
            echo "[Node $((NODE_INDEX + 1)), Sim ${{sim_on_node}}] Starting restart job ${{job_id}}: ${{airfoil_name}} at Re=${{reynolds}} on ${{NODE}}"
            
            # Launch the restart simulation
            (
                cd "${{work_dir}}" || {{
                    echo "[Job ${{job_id}}] ERROR: Cannot access directory ${{work_dir}}" | tee -a "${{STATUS_FILE}}"
                    echo "${{job_id}}|FAILED|Cannot access directory" >> "${{COMPLETED_FILE}}"
                    exit 1
                }}

                echo "[Job ${{job_id}}] Linking files from initial run..." >> "${{STATUS_FILE}}"

                # Save current path to initial_dir and return later
                initial_abs=$(cd "${{initial_dir}}" && pwd)

                # Recreate directory structure
                cd "${{initial_abs}}"
                rm "${{airfoil_name}}0.f00001"
                find . -type d | while read dir; do mkdir -p "${{work_dir}}/$dir"; done
                find . -type f ! -name '*.par' | while read file; do ln -s "${{initial_abs}}/$file" "${{work_dir}}/$file"; done
                find . -type f -name '*.par' | while read file; do cp "${{initial_abs}}/$file" "${{work_dir}}/$file"; done
                # Return to work_dir (important!)
                cd "${{work_dir}}"

                # Update par file for restart
                echo "[Job ${{job_id}}] Updating par file for restart at Re=${{reynolds}}..." >> "${{STATUS_FILE}}"
                python3 << EOF
import re

par_file = '${{airfoil_name}}.par'
with open(par_file, 'r') as f:
    content = f.read()

# Update for restart
content = re.sub(r'-10000.0', '-${{reynolds}}', content)
content = re.sub(r'-10000', '-${{reynolds}}', content)
content = re.sub(r'dt = 5.0e-7', 'dt = 5.0e-7', content)
content = re.sub(r'#startFrom = rans0.f00002', 'startFrom = ${{airfoil_name}}0.f00002', content)
content = re.sub(r'#timeStepper = BDF2', 'timeStepper = BDF2', content)
content = re.sub(r'#extrapolation = OIFS', 'extrapolation = OIFS', content)
content = re.sub(r'#targetCFL = 3.5.', 'targetCFL = 3.5.', content)
content = re.sub(r'numsteps = 100000', 'numsteps = 1000000', content)
content = re.sub(r'writeInterval = 100000', 'writeInterval = 1000000', content)

with open(par_file, 'w') as f:
    f.write(content)
EOF
                
                # Restart run
                echo "[Job ${{job_id}}] Starting restart run at Re=${{reynolds}}..." >> "${{STATUS_FILE}}"
                echo ${{airfoil_name}} > SESSION.NAME
                echo $(pwd)'/' >> SESSION.NAME 

                # Create a hostfile for this specific simulation
                echo "${{NODE}}" > hostfile_job_${{job_id}}

                MPI_ARG="-n ${{NRANKS_PER_NODE}} --ppn ${{NRANKS_PER_NODE}} --depth=${{NDEPTH}} --cpu-bind depth"


                mpiexec ${{MPI_ARG}} --hostfile hostfile_job_${{job_id}} ./nek5000 $airfoil_name > /dev/null 2>&1

                if [ $? -ne 0 ]; then
                    echo "[Job ${{job_id}}] ERROR: Restart run failed for ${{airfoil_name}}" | tee -a "${{STATUS_FILE}}"
                    echo "${{job_id}}|FAILED|Restart run failed" >> "${{COMPLETED_FILE}}"
                    rm -f hostfile_job_${{job_id}}
                    exit 1
                fi              

                echo "[Job ${{job_id}}] Restart simulation completed successfully: ${{airfoil_name}} at Re=${{reynolds}}" | tee -a "${{STATUS_FILE}}"
                echo "${{job_id}}|SUCCESS|Completed" >> "${{COMPLETED_FILE}}"
                
                # Clean up hostfile
                rm -f hostfile_job_${{job_id}}
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

# Stop save_memory.sh
kill $SAVE_MEMORY_PID 2>/dev/null || true

echo "=============================================="
echo "All restart simulations completed"
echo "=============================================="
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
            print(result.stdout)
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
        # Keep the actual angle values for velocity calculation
        reynolds_to_angles[reynolds] = sorted(angles_rad)
        all_angles.update(angles_rad)

    return reynolds_to_angles, sorted(all_angles)

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
    # Format: multiply by 100 and round to get unique names like 1135 for 11.35
    angle_dir = str(int(round(angle * 100)))
    if angle < 0:
        angle_dir = "neg" + angle_dir[1:]  # Remove minus sign and prepend "neg"
    initial_dir = INITIAL_RANS_RUNS_DIR / airfoil_name / angle_dir
    initial_dir.mkdir(parents=True, exist_ok=True)
    return initial_dir

def create_restart_directory(airfoil_name, angle, reynolds):
    """Create directory structure for restart RANS run."""
    # Use floor for directory names
    angle_dir = str(int(np.floor(angle))).replace("-", "neg")
    restart_dir = RESTART_RANS_RUNS_DIR / airfoil_name / f"Re{int(reynolds)}" / angle_dir
    restart_dir.mkdir(parents=True, exist_ok=True)
    return restart_dir

def modify_usr_file_for_angle(usr_file, angle_deg):
    """Modify the rans.usr file to set velocity components based on angle."""
    # Calculate velocity components
    angle_rad = math.radians(angle_deg)
    ux_value = math.cos(angle_rad)
    uy_value = math.sin(angle_rad)
    
    # Read the usr file
    with open(usr_file, 'r') as f:
        content = f.read()
    
    # Replace ux=1.0 with ux=cos(angle)
    content = re.sub(r'ux\s*=\s*1\.0', f'ux={ux_value:.16f}', content)
    
    # Replace uy=0.0 with uy=sin(angle)
    content = re.sub(r'uy\s*=\s*0\.0', f'uy={uy_value:.16f}', content)
    
    # Write back the modified content
    with open(usr_file, 'w') as f:
        f.write(content)
    
    print(f"  Modified rans.usr: ux={ux_value:.6f}, uy={uy_value:.6f} (angle={angle_deg:.2f}°)")

def prepare_initial_simulation_files(initial_dir, airfoil_name, angle_deg):
    """Prepare files for initial simulation at Re=10000."""
    # Copy all files from rans_base
    if (initial_dir / 'nek5000').exists():
            for file in RANS_BASE_DIR.iterdir():
                if file.is_file() and file.name == "rans.par":
                    shutil.copy2(file, initial_dir)
            old_par = initial_dir / "rans.par"
            new_par = initial_dir / f"{airfoil_name}.par"
            if old_par.exists():
                shutil.copy2(old_par, new_par)
                old_par.unlink()
            genmap_input_file = initial_dir / "genmap_input.txt"
            with open(genmap_input_file, 'w') as f:
                f.write(f"{airfoil_name}\n0.05\n")
            if (initial_dir / f"{airfoil_name}0.f00002").exists():
                return None
            return airfoil_name

    for file in RANS_BASE_DIR.iterdir():
        if file.is_file():
            shutil.copy2(file, initial_dir)

    # Copy RE2 file (use unrotated mesh)
    re2_source = RE2_FILES_DIR / f"{airfoil_name}.re2"
    shutil.copy2(re2_source, initial_dir)

    # Modify the usr file for the angle
    usr_file = initial_dir / "rans.usr"
    modify_usr_file_for_angle(usr_file, angle_deg)

    # Compile the modified code
    print(f"  Compiling for angle {angle_deg:.2f}°...")
    compile_result = run_command(["make"], cwd=initial_dir, check=False)
    if compile_result.returncode != 0:
        print(f"ERROR: Compilation failed in {initial_dir}")
        print(compile_result.stderr)
        sys.exit(1)

    # Create genmap input file
    genmap_input_file = initial_dir / "genmap_input.txt"
    with open(genmap_input_file, 'w') as f:
        f.write(f"{airfoil_name}\n0.05\n")

    # Prepare initial par file (with Re=10000)
    old_par = initial_dir / "rans.par"
    new_par = initial_dir / f"{airfoil_name}.par"
    shutil.copy2(old_par, new_par)
    old_par.unlink()

    return airfoil_name  # Always return the base airfoil name

def create_batch_jobs(initial_job_infos, restart_job_infos, num_nodes=None, 
                     walltime=None, queue="workq-route"):
    """Create PBS job scripts for initial and restart runs."""
    if num_nodes is None:
        num_nodes = DEFAULT_NODES
    
    if walltime is None:
        walltime = DEFAULT_WALLTIME
    
    sims_per_node = CORES_PER_NODE // TASKS_PER_SIMULATION
    max_parallel_sims = num_nodes * sims_per_node
    
    job_files = []
    
    # Create initial runs job if needed
    if initial_job_infos:
        print(f"\nCreating PBS job for {len(initial_job_infos)} initial simulations")
        print(f"  Nodes: {num_nodes}")
        print(f"  Queue: {queue}")
        print(f"  Walltime: {walltime}")
        
        # Create job data string for initial runs
        job_data_lines = []
        for i, job_info in enumerate(initial_job_infos):
            line = f"{i}|{job_info['work_dir']}|{job_info['airfoil_name']}"
            job_data_lines.append(line)
        
        jobs_data = '\n'.join(job_data_lines)
        
        # Create initial job script
        batch_id = "1"
        job_script = PBS_INITIAL_TEMPLATE.format(
            batch_id=batch_id,
            queue=queue,
            num_nodes=num_nodes,
            walltime=walltime,
            jobs_data=jobs_data,
            total_jobs=len(initial_job_infos),
            pbs_jobs_dir=str(PBS_JOBS_DIR),
            tasks_per_simulation=TASKS_PER_SIMULATION,
            cores_per_node=CORES_PER_NODE
        )
        
        # Save job script
        job_file = PBS_JOBS_DIR / f"rans_initial_{batch_id}.pbs"
        with open(job_file, 'w') as f:
            f.write(job_script)
        
        job_file.chmod(0o755)
        job_files.append(('initial', job_file))
    
    # Create restart runs job if needed
    if restart_job_infos:
        print(f"\nCreating PBS job for {len(restart_job_infos)} restart simulations")
        
        # Create job data string for restart runs
        job_data_lines = []
        for i, job_info in enumerate(restart_job_infos):
            line = f"{i}|{job_info['work_dir']}|{job_info['airfoil_name']}|{job_info['reynolds']}|{job_info['initial_dir']}"
            job_data_lines.append(line)
        
        jobs_data = '\n'.join(job_data_lines)
        
        # Create restart job script
        batch_id = "1"
        job_script = PBS_RESTART_TEMPLATE.format(
            batch_id=batch_id,
            queue=queue,
            num_nodes=num_nodes,
            walltime=walltime,
            jobs_data=jobs_data,
            total_jobs=len(restart_job_infos),
            pbs_jobs_dir=str(PBS_JOBS_DIR),
            tasks_per_simulation=TASKS_PER_SIMULATION,
            cores_per_node=CORES_PER_NODE
        )
        
        # Save job script
        job_file = PBS_JOBS_DIR / f"rans_restart_{batch_id}.pbs"
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

    # First: Convert airfoil to RE2 (only once, no rotation)
    print("\nPhase 1: Converting airfoil to RE2...")
    convert_to_re2(airfoil_name)

    # Second: Prepare initial runs (one per angle)
    print("\nPhase 2: Preparing initial runs at Re=10000...")
    for angle_deg in all_angles:
        initial_dir = create_initial_directory(airfoil_name, angle_deg)
        airfoil_name_used = prepare_initial_simulation_files(initial_dir, airfoil_name, angle_deg)
        angle_to_initial_dir[angle_deg] = initial_dir
        
        if airfoil_name_used == None:
            continue
        
        initial_job_info = {
            'airfoil': airfoil_name,
            'angle': angle_deg,
            'airfoil_name': airfoil_name_used,
            'work_dir': str(initial_dir)
        }
        
        initial_job_infos.append(initial_job_info)
        print(f"  Prepared initial simulation: {airfoil_name_used} at {angle_deg:.2f}°")

    # Third: Prepare restart runs (for all reynolds numbers)
    print("\nPhase 3: Preparing restart runs...")
    for reynolds, angles in reynolds_to_angles.items():
        # Skip if this is the initial Reynolds number
        if reynolds == INITIAL_REYNOLDS:
            continue
            
        for angle_deg in angles:
            restart_dir = create_restart_directory(airfoil_name, angle_deg, reynolds)
            
            restart_job_info = {
                'airfoil': airfoil_name,
                'angle': angle_deg,
                'reynolds': reynolds,
                'airfoil_name': airfoil_name,
                'work_dir': str(restart_dir),
                'initial_dir': str(angle_to_initial_dir[angle_deg])
            }
            
            restart_job_infos.append(restart_job_info)
            print(f"  Prepared restart simulation: {airfoil_name} at {angle_deg:.2f}° and Re={reynolds}")

    return initial_job_infos, restart_job_infos

def submit_jobs(job_files, submit_restart_after_initial=True):
    """Submit PBS jobs with optional dependency."""
    job_ids = {}
    
    for job_type, job_file in job_files:
        cmd = ["qsub"]
        
        # If this is a restart job and we have an initial job, add dependency
        if job_type == 'restart' and 'initial' in job_ids and submit_restart_after_initial:
            cmd.extend(["-W", f"depend=afterany:{job_ids['initial']}"])
        elif job_type == 'restart':
            continue
        cmd.append(str(job_file))
        
        result = run_command(cmd, check=False)
        
        if result and result.returncode == 0:
            job_id = result.stdout.strip()
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
    print("Optimized Parallelized RANS Simulation Automation Script for Polaris")
    print("Modified to rotate velocity instead of airfoil mesh")
    print("===================================================================")

    # Create necessary directories
    INITIAL_RANS_RUNS_DIR.mkdir(exist_ok=True)
    RESTART_RANS_RUNS_DIR.mkdir(exist_ok=True)
    PBS_JOBS_DIR.mkdir(exist_ok=True)

    # Load CSV data
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    all_initial_job_infos = []
    all_restart_job_infos = []

    # Prepare all simulations
    for airfoil_name in AIRFOILS_TO_PROCESS:
        if check_airfoil_exists(airfoil_name, df):
            initial_jobs, restart_jobs = prepare_all_simulations(airfoil_name, df)
            if initial_jobs == None and restart_jobs == None:
                continue
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
            print("  2. workq-route (1-184 nodes, 5min-24hr)")
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

        # Create batch jobs
        batch_files = create_batch_jobs(all_initial_job_infos, all_restart_job_infos,
                                       num_nodes=num_nodes, walltime=walltime, queue=queue)
        
        print(f"\nBatch job configuration:")
        if all_initial_job_infos:
            print(f"  - {len(all_initial_job_infos)} initial simulations")
        if all_restart_job_infos:
            print(f"  - {len(all_restart_job_infos)} restart simulations")
        print(f"  - {num_nodes} nodes per job")
        print(f"  - Queue: {queue}")
        print(f"  - Walltime: {walltime}")

        # Ask user about job submission
        response = input("\nSubmit batch jobs to PBS? (y/n): ").lower()
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
            print("Monitor progress with: qstat -u $USER")
            print(f"Check outputs in: {PBS_JOBS_DIR}/")
            
            if 'initial' in job_ids:
                print(f"\nInitial runs:")
                print(f"  - Output: rans_initial_1.o")
                print(f"  - Errors: rans_initial_1.e")
                print(f"  - Status: initial_status_1.txt")
                print(f"  - Results: initial_completed_1.txt")
            
            if 'restart' in job_ids:
                print(f"\nRestart runs:")
                print(f"  - Output: rans_restart_1.o")
                print(f"  - Errors: rans_restart_1.e")
                print(f"  - Status: restart_status_1.txt")
                print(f"  - Results: restart_completed_1.txt")
            
            print(f"{'='*60}")
        else:
            print("\nBatch job scripts created but not submitted.")
            print(f"To submit manually:")
            for job_type, job_file in batch_files:
                print(f"  qsub {job_file}")
            print(f"Scripts are in: {PBS_JOBS_DIR}")
    else:
        print("\nNo simulations to run.")

if __name__ == "__main__":
    main()
