import os
import pandas as pd


def read_coordinates(file_path):
    with open(file_path, 'r') as f:
        coords = []
        for line in f:
            try:
                x, y = map(float, line.strip().split())
                coords.append((x, y))
            except ValueError:
                continue
        return coords


def parse_all_pd(file_path, coordinates_dir):
    airfoil_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            
            if lines[i].startswith("Airfoil"):
                airfoil_name = lines[i].split()[1]
                builder = lines[i + 1].split()[1]  
                num_reynolds = int(lines[i + 2].strip())  
                
                
                coord_file = os.path.join(coordinates_dir, f"{airfoil_name}.cor")
                if not os.path.exists(coord_file):
                    i += 3 + num_reynolds * 2  
                    continue  
                
                coordinates = read_coordinates(coord_file)
                
                
                i += 3
                for _ in range(num_reynolds):
                    reynolds_number = float(lines[i].strip())
                    num_tests = int(lines[i + 1].strip())
                    i += 2
                    for _ in range(num_tests):
                        
                        print(lines[i].strip())
                        lift, drag, angle = map(float, lines[i].strip().split())
                        airfoil_data.append({
                            "airfoil_name": airfoil_name,
                            "coordinates": coordinates,
                            "reynolds_number": reynolds_number,
                            "angle_of_attack": angle,
                            "lift_coefficient": lift,
                            "drag_coefficient": drag
                        })
                        i += 1
            else:
                i += 1
    return airfoil_data


all_pd_path = 'Stec8/ALL.PD'
coordinates_dir = 'Stec8/'


parsed_data = parse_all_pd(all_pd_path, coordinates_dir)


df = pd.DataFrame(parsed_data)

output_path = 'training_data_stec8.csv'
df.to_csv(output_path, index=False)
