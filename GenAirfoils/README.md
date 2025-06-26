Usage Examples for test_gen.py:

```bash
pip install gmsh
python test_gen.py --list                  # List available files
python test_gen.py --airfoil NACA0012      # Specific airfoil
python test_gen.py --all                   # Process all airfoils
python test_gen.py --samples 20            # 20 random samples
```

How to use rotate_dat.py:  
Default database (input) is './airfoil_database/'  
Default output directory is './airfoil_database/'  
```bash
python3 rotate_dat.py --list                                                                    # List all available airfoils 
python3 rotate_dat.py --all 15                                                                  # Rotate all airfoils by 15 degrees
python3 rotate_dat.py --airfoil a18 30                                                          # Rotate a specific airfoil by 30 degrees
python3 rotate_dat.py --all 45 --database ./my_airfoils/ --output ./airfoil_rot_database/       # Use custom directories

python3 rotate_dat.py --all 45 --output ./airfoil_rot_database/                                 # Most common command for testing
```

Error files:
- ah88k136 (goes through, but warning)
- ah93w480b (doesn't go through)
- boe106 (doesn't go through)
- cap21c (doesn't go through)
- dsma523a (doesn't go through)
- e337 (goes through, but warning)
- e338 (goes through, but warning)
- e378 (doesn't go through)
- e49 (no warning, just doesn't finish)
- e817 (doesn't go through)
- e856 (goes through, but warning)
- fx62k131 (doesn't go through)
- fx63147 (goes through, but lots of problems)
- fx69h083 (goes through, but warning)
- fx76mp120 (no such file)
- fx77w270 (goes through, but warning)
- fx79w470a (goes through, but warning)
- goe113 (goes through, but warning)
- goe123 (goes through, but warning)
- goe300 (goes through, but warning)
- goe346 (no such file?)
- goe744 (no such file)
- hq300gd2 (goes through, but warning)
- m8 (goes through, but warning)
- m9 (goes through, but warning)
- mh150 (no warning, just doesn't finish)
- mh60 (goes through, but warning)
- naca63206 (goes through, but warning)
- naca63a210 (goes through, but warning)
- rc0864c (goes through, but warning)
- s4158 (no such file)
- strand (doesn't go through)
- usa27 (no warning, just doesn't finish)
- usa27m2 (no warning, just doesn't finish)
- usa41 (no such file)
- wb140 (goes through, but warning)