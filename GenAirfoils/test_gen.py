import subprocess
import os, random
import numpy as np
import shutil


class MeshCreationError(Exception):
    pass

samples           = 10           # no. of datasets to produce

airfoil_database  = "./airfoil_database/"
output_dir        = "./re2_files/"
os.makedirs(output_dir, exist_ok=True)

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))

def genMesh(airfoilFile):
    airfoilName = airfoilFile.split('.')[1].split('/')[2]
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[-1]))<1e-6:
        ar = ar[:-1]

    if np.abs(ar[0][1]+ar[-1][1]) < 1e-6:
        ar = ar[:-1]
        ar[0][1] = 0.

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                outFile.write(line)

    if os.system("gmsh airfoil.geo -2 -format msh2 -order 2 -o airfoil.msh  > /dev/null") != 0:
        raise MeshCreationError('GMSH failed to create mesh')

    print("GMSH complete")

    responses = ["2", "airfoil", "0", "0", airfoilName]
    input_data = "\n".join(responses) + "\n"
    process = subprocess.Popen(
        ["gmsh2nek"],
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE,  
        stderr=subprocess.PIPE,  
        text=True
    )
    stdout, stderr = process.communicate(input=input_data)
    print("NEK complete")
    try:
        shutil.move(airfoilName + ".re2", output_dir)  
    except shutil.Error:
        shutil.rmtree(airfoilName + '.re2')


    return(0)

files = os.listdir(airfoil_database)
files.sort()
if len(files)==0:
	print("error - no airfoils found in %s" % airfoil_database)
	exit(1)

# main
for n in range(samples):
    print("Run {}:".format(n))

    fileNumber = np.random.randint(0, len(files))
    basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]

    print("\tusing {}".format(files[fileNumber]))

    genMesh(airfoil_database + files[fileNumber])