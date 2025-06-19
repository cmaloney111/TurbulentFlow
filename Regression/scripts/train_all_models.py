#!/usr/bin/env python3
"""
Train all regression models
"""
import sys
import os
import subprocess

def run_script(script_path, description):
    print(f"\n{'='*50}")
    print(f"Training {description}...")
    print(f"{'='*50}")
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} training completed successfully")
    else:
        print(f"✗ {description} training failed")
    
    return result.returncode

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'models')
    
    models = [
        # ('random_forest.py', 'Random Forest'),
        ('xgboost_model.py', 'XGBoost'),
        # ('neuralfoil_ann.py', 'Neural Network'),
        # ('train_blind_nn.py', 'Neural Network')
    ]
    
    success_count = 0
    
    for script, description in models:
        script_path = os.path.join(base_dir, script)
        if os.path.exists(script_path):
            if run_script(script_path, description) == 0:
                success_count += 1
        else:
            print(f"Warning: {script} not found")
    
    print(f"\n{'='*50}")
    print(f"Training Summary: {success_count}/{len(models)} models trained successfully")
    print(f"{'='*50}")
    
    return 0 if success_count == len(models) else 1

if __name__ == '__main__':
    sys.exit(main())