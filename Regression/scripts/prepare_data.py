#!/usr/bin/env python3
"""
Data preparation script - extracts and processes Stec8 data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_processing'))

from make_csv import parse_all_pd
import pandas as pd

def main():
    print("Preparing training data from Stec8 database...")
    
    # Extract data from Stec8
    all_pd_path = 'Stec8/ALL.PD'
    coordinates_dir = 'Stec8/'
    
    if not os.path.exists(all_pd_path):
        print(f"Error: {all_pd_path} not found")
        return 1
    
    parsed_data = parse_all_pd(all_pd_path, coordinates_dir)
    df = pd.DataFrame(parsed_data)
    
    output_path = 'training_data_stec8.csv'
    df.to_csv(output_path, index=False)
    print(f"Training data saved to {output_path}")
    print(f"Dataset contains {len(df)} samples from {df['airfoil_name'].nunique()} airfoils")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())