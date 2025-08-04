import os
import re
import sys

max_all = 0
min_all = 50000
max_logpath = ""
min_logpath = ""
def find_last_step(log_path):
    global max_all, max_logpath, min_all, min_logpath
    try:
        with open(log_path, "r") as f:
            steps = [int(m.group(1)) for m in re.finditer(r"Step\s+(\d+),", f.read())]
            max_step = max(steps) if steps else None
            if max_step > max_all:
                max_all = max_step
                max_logpath = log_path
            if max_step < min_all:
                min_all = max_step
                min_logpath = log_path
            return max_step
    except Exception:
        return None

def scan_all(base_dir, output_file):
    with open(output_file, "w") as out:
        out.write("airfoil,reynolds,angle,step\n")
        for airfoil in sorted(os.listdir(base_dir)):
            airfoil_path = os.path.join(base_dir, airfoil)
            if not os.path.isdir(airfoil_path):
                continue
            for reynolds in sorted(os.listdir(airfoil_path)):
                reynolds_path = os.path.join(airfoil_path, reynolds)
                if not os.path.isdir(reynolds_path):
                    continue
                for angle in sorted(os.listdir(reynolds_path)):
                    angle_path = os.path.join(reynolds_path, angle)
                    if not os.path.isdir(angle_path):
                        continue
                    for fname in os.listdir(angle_path):
                        if fname == 'nek_restart.log':
                            log_path = os.path.join(angle_path, fname)
                            step = find_last_step(log_path)
                            out.write(f"{airfoil},{reynolds},{angle},{step}\n")
                            break

scan_all("restart_rans_runs", "output.csv")
print(f"Highest step: {max_all} at {max_logpath}")
print(f"Lowest step: {min_all} at {min_logpath}")
