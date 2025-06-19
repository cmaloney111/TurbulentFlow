#!/usr/bin/env python3
import pandas as pd

# Load the training data
df = pd.read_csv('training_data_stec8.csv')

print('Dataset Overview:')
print(f'Total rows: {len(df)}')
print(f'Unique airfoils: {len(df["airfoil_name"].unique())}')
print(f'Unique Reynolds numbers: {sorted(df["reynolds_number"].unique())}')
print()

print('Reynolds numbers by airfoil (first 15 airfoils):')
airfoil_reynolds = {}
for airfoil in sorted(df['airfoil_name'].unique())[:15]:
    reynolds = sorted(df[df['airfoil_name'] == airfoil]['reynolds_number'].unique())
    airfoil_reynolds[airfoil] = reynolds
    print(f'{airfoil}: {reynolds}')

print()
print('Sample of airfoils with different Reynolds number coverage:')
for airfoil, reynolds_list in list(airfoil_reynolds.items())[:5]:
    print(f'{airfoil} has {len(reynolds_list)} Reynolds numbers: {reynolds_list}')