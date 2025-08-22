# TurbulentFlow
Everything is based on running from the Polaris Supercomputer from the ALCF at Argonne National Laboratory

# To download Nek5000
```
cd ~
wget https://github.com/Nek5000/Nek5000/archive/refs/heads/master.zip
unzip master.zip
mv Nek5000-master Nek5000
rm master.zip
export PATH=$HOME/Nek5000/bin:$PATH
cd ~/Nek5000/tools
module load PrgEnv-gnu
```
Polaris supercomputer had an issue with it being called gfortran so we had to create a symlink, Can ignore if you have gfortran :
```
	mkdir -p ~/bin
    ln -sf /usr/bin/gfortran-13 ~/bin/gfortran
    export PATH=~/bin:$PATH
```
The last part of downloading Nek5000
```
cd ~/Nek5000/tools
./maketools genmap
```

I added this to the ~/.bash_profile for simplicity:
```
PATH=$PATH:$HOME/.local/bin:$HOME/bin:$HOME/Nek5000/bin
module swap PrgEnv-nvhpc PrgEnv-gnu

export CC="mpicc"
export FC="mpif90"
export FCFLAGS="-fPIC"
export FFLAGS="-fPIC"
export CFLAGS="-fPIC"
export CXXFLAGS="-fPIC"
export LDFLAGS="-Wl,--no-relax"
export PATH

export LD_LIBRARY_PATH=/opt/cray/pe/ccdb/5.0.2/lib:$LD_LIBRARY_PATH
```
^ you may need to change this library path for your own setup, but this is the path for Polaris
# TO TEST NEK5000
```
cd ~
wget https://github.com/Nek5000/NekExamples/archive/refs/heads/master.zip
unzip master.zip
mv NekExamples-master NekExamples
rm master.zip
cp -r ~/NekExamples/eddy_uv .
cd eddy_uv
genmap              # on input type eddy_uv, then enter for default 
makenek eddy_uv             	
./nek5000			# works always, default to this if no gpu
nekbmpi eddy_uv 2	# run this if you have a gpu
tail logfile        
visnek eddy_uv      # produces the eddy_uv.nek5000 file
```


You can scp it onto your actual machine and run to visualize data
```
visit -o eddy_uv.nek5000
```

For some reason SESSION.NAME 	was missing sometimes. So here it is:
```
eddy_uv
/home/<username>/eddy_uv/
```
^ Should just be your working directory


These folders are not needed anymore:
```
cd ..
rm -r eddy_uv/
rm -r NekExamples/
```



# TO DOWNLOAD THE PROJECT
```
cd ~
git clone https://github.com/cmaloney111/TurbulentFlow.git
cd TurbulentFlow/rans/
python -m pip install -r requirements.txt
```

To test just type 
```
gmsh
```

If you got a libglu error follow the steps below, if nothing happened then it worked.

# To load ligblu for POLARIS
```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
source ~/.bash_profile
conda install -c conda-forge libglu
```

# TO RUN 
```
vim run_rans.py       	#change airfoils_to_process to whichever ones you want
python run_rans.py
```
