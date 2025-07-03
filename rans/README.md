Steps:

1. run "pip install -r requirements.txt"
2. Update run_rans.py and put your specific airfoils in AIRFOILS_TO_PROCESS
3. update ~/Nek5000/bin/nekmpi to have what's below
3. run "python run_rans.py"


nekmpi:
```bash
#!/bin/bash
echo $1        >  SESSION.NAME
echo `pwd`'/' >>  SESSION.NAME
rm -f logfile
rm -f ioinfo
mv $1.log.$2 $1.log1.$2 2>/dev/null
mpiexec -np $2 ./nek5000 > logfile
```
