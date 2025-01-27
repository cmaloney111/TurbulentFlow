import subprocess

import os, math, random
import numpy as np
import shutil;


samples           = 100           # no. of datasets to produce
freestream_angle  = math.pi / 8.  # -angle ... angle
freestream_length = 10.           # len * (1. ... factor)
freestream_length_factor = 10.    # length factor

airfoil_database  = "./airfoil_database/"
output_dir        = "./test_output/"

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))

def genMesh(airfoilFile):
    airfoilName = airfoilFile.split('.')[1].split('/')[2]
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
        ar = ar[:-1]

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
        print("error during mesh creation!")
        return(-1)


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
    try:
        shutil.move(airfoilName + ".re2", "output_test")  
    except shutil.Error:
        shutil.rmtree(airfoilName + '.re2')
    


    return(0)

def runSim(freestreamX, freestreamY):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)

    os.system("./Allclean && simpleFoam > foam.log")

def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, pfile='OpenFOAM/postProcessing/internalCloud/500/cloud_p.xy', ufile='OpenFOAM/postProcessing/internalCloud/500/cloud_U.xy', res=128, imageIndex=0): 
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    npOutput = np.zeros((6, res, res))

    ar = np.loadtxt(pfile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[3][x][y] = ar[curIndex][3]
                curIndex += 1
                # fill input as well
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
            else:
                npOutput[3][x][y] = 0
                # fill mask
                npOutput[2][x][y] = 1.0

    ar = np.loadtxt(ufile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[4][x][y] = ar[curIndex][3]
                npOutput[5][x][y] = ar[curIndex][4]
                curIndex += 1
            else:
                npOutput[4][x][y] = 0
                npOutput[5][x][y] = 0

    # utils.saveAsImage('data_pictures/pressure_%04d.png'%(imageIndex), npOutput[3])
    # utils.saveAsImage('data_pictures/velX_%04d.png'  %(imageIndex), npOutput[4])
    # utils.saveAsImage('data_pictures/velY_%04d.png'  %(imageIndex), npOutput[5])
    # utils.saveAsImage('data_pictures/inputX_%04d.png'%(imageIndex), npOutput[0])
    # utils.saveAsImage('data_pictures/inputY_%04d.png'%(imageIndex), npOutput[1])

    #fileName = dataDir + str(uuid.uuid4()) # randomized name
    fileName = dataDir + "%s_%d_%d" % (basename, int(freestreamX*100), int(freestreamY*100) )
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)


files = os.listdir(airfoil_database)
files.sort()
if len(files)==0:
	print("error - no airfoils found in %s" % airfoil_database)
	exit(1)

# utils.makeDirs( ["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"] )


# main
for n in range(samples):
    print("Run {}:".format(n))

    fileNumber = np.random.randint(0, len(files))
    basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]
    print("\tusing {}".format(files[fileNumber]))

    length = freestream_length * np.random.uniform(1.,freestream_length_factor) 
    angle  = np.random.uniform(-freestream_angle, freestream_angle) 
    fsX =  math.cos(angle) * length
    fsY = -math.sin(angle) * length

    print("\tUsing len %5.3f angle %+5.3f " %( length,angle )  )
    print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))

    if genMesh(airfoil_database + files[fileNumber]) != 0:
        print("\tmesh generation failed, aborting")
        continue
    break

    
    # runSim(fsX, fsY)
    # os.chdir("..")

    # outputProcessing(basename, fsX, fsY, imageIndex=n)
    # print("\tdone")