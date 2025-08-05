import os
import re
import sys

max_all = 0
min_all = float("inf")
max_logpath = ""
min_logpath = ""

def find_last_step(log_path):
    global max_all, max_logpath, min_all, min_logpath
    try:
        with open(log_path, "r") as f:
            steps = [int(m.group(1)) for m in re.finditer(r"Step\s+(-?\d+),", f.read())]
            if not steps:
                return None
            max_step = max(steps)
            if max_step == -1:
                return max_step
            if max_step > max_all:
                max_all = max_step
                max_logpath = log_path
            if max_step < min_all:
                min_all = max_step
                min_logpath = log_path
            return max_step
    except Exception:
        return None

def scan_initial_runs(base_dir, output_file):
    with open(output_file, "w") as out:
        out.write("airfoil,angle,step\n")
        if len(sys.argv) == 1:
            airfoils = sorted(os.listdir(base_dir))
        else:
            airfoils = sys.argv[1:]
        for airfoil in airfoils:
            airfoil_path = os.path.join(base_dir, airfoil)
            if not os.path.isdir(airfoil_path):
                continue
            for angle in sorted(os.listdir(airfoil_path)):
                angle_path = os.path.join(airfoil_path, angle)
                if not os.path.isdir(angle_path):
                    continue
                for fname in os.listdir(angle_path):
                    if fname == "nek_initial.log":
                        log_path = os.path.join(angle_path, fname)
                        step = find_last_step(log_path)
                        out.write(f"{airfoil},{angle},{step}\n")
                        break  # Skip other log files in the same folder

# Change this if needed
scan_initial_runs("initial_rans_runs", "output.csv")

print(f"Highest step: {max_all} at {max_logpath}")
print(f"Lowest step: {min_all} at {min_logpath}")

