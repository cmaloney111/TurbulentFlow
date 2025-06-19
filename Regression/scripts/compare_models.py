#!/usr/bin/env python3
"""
Interactive model comparison script
"""
import sys
import os
import subprocess

def main():
    print("Starting interactive model comparison...")
    print("Available models: rff, xgb, ann, net")
    
    # Change to the project directory
    project_dir = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(project_dir)
    
    try:
        result = subprocess.run([sys.executable, 'src/analysis/test_models.py'], 
                              check=True,
                              text=True,
                              stdin=sys.stdin,
                              stdout=sys.stdout,
                              stderr=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        print("Error: test_models.py not found")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Error running model comparison: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())