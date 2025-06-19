#!/usr/bin/env python3
"""
Train all streamlit models in non-verbose mode.
This script uses the model_utils.py functions to train all models
with the same parameters and file naming conventions used by the streamlit app.
"""

import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import model training functions from model_utils
from model_utils import (
    load_training_data, 
    prepare_data, 
    train_random_forest_model,
    train_xgboost_model, 
    train_ann_model, 
    train_neuralfoil_model
)

def create_models_dir():
    """Create models directory if it doesn't exist"""
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
        print(f"Created models directory: {models_dir}")

def check_training_data():
    """Check if training data file exists"""
    training_data_file = Path("training_data_stec8.csv")
    if not training_data_file.exists():
        print(f"❌ Training data file not found: {training_data_file}")
        print("Please ensure training_data_stec8.csv exists in the current directory.")
        return False
    return True

def train_model_with_timing(model_name, train_func, *args, **kwargs):
    """Train a model and measure the time taken"""
    print(f"🔄 Training {model_name}...")
    start_time = time.time()
    
    try:
        model, results = train_func(*args, verbose=False, **kwargs)
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✓ {model_name} completed in {training_time:.1f}s")
        print(f"  MAE: {results['mae']:.4f} | MSE: {results['mse']:.4f} | R²: {results['r2']:.4f}")
        return True, results, training_time
        
    except Exception as e:
        end_time = time.time()
        training_time = end_time - start_time
        print(f"✗ {model_name} failed after {training_time:.1f}s: {str(e)}")
        return False, {"error": str(e)}, training_time

def main():
    """Train all streamlit models in non-verbose mode"""
    print("🚀 Training all Streamlit models in non-verbose mode...")
    print("=" * 70)
    
    # Check prerequisites
    if not check_training_data():
        return False
    
    create_models_dir()
    
    # Load and prepare data once for all models
    print("📊 Loading and preparing training data...")
    try:
        data = load_training_data()
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            data, 
            test_size=0.2,
            random_state=42,
            use_scaling=True
        )
        print(f"✓ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return False
    
    print("\n" + "=" * 70)
    
    # Define models to train with their parameters
    models_to_train = [
        {
            "name": "Random Forest",
            "function": train_random_forest_model,
            "params": {
                "n_estimators": 10,  # Minimal for fast testing
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": None,
                "random_state": 42
            }
        },
        {
            "name": "XGBoost",
            "function": train_xgboost_model,
            "params": {
                "n_estimators": 10,  # Minimal for fast testing
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0.0,
                "random_state": 42
            }
        },
        {
            "name": "ANN (Neural Network)",
            "function": train_ann_model,
            "params": {
                "epochs": 5,  # Minimal for fast testing
                "batch_size": 32,
                "learning_rate": 0.01,
                "layers": "64,32",  # Smaller network for fast testing
                "activation": "relu"
            }
        },
        {
            "name": "NeuralFoil",
            "function": train_neuralfoil_model,
            "params": {
                "epochs": 5,  # Minimal for fast testing
                "batch_size": 32,
                "learning_rate": 1e-3,
                "hidden_layers": 2,
                "width": 32
            }
        }
    ]
    
    # Train each model
    results = {}
    total_start_time = time.time()
    
    for i, model_config in enumerate(models_to_train, 1):
        model_name = model_config["name"]
        train_func = model_config["function"]
        params = model_config["params"]
        
        success, model_results, training_time = train_model_with_timing(
            model_name, train_func,
            X_train, y_train, X_test, y_test, scaler,
            **params
        )
        
        results[model_name] = {
            "success": success,
            "results": model_results,
            "training_time": training_time
        }
        
        print()  # Add spacing between models
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Summary
    print("=" * 70)
    print("📊 Training Summary:")
    print(f"⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    
    successful_models = []
    failed_models = []
    
    for model_name, result in results.items():
        if result["success"]:
            successful_models.append(model_name)
            model_results = result["results"]
            training_time = result["training_time"]
            print(f"✓ {model_name} ({training_time:.1f}s)")
            print(f"  MAE: {model_results['mae']:.4f} | MSE: {model_results['mse']:.4f} | R²: {model_results['r2']:.4f}")
        else:
            failed_models.append(model_name)
            print(f"✗ {model_name}: {result['results']['error']}")
    
    if successful_models:
        print(f"\n✅ Successfully trained {len(successful_models)}/{len(models_to_train)} models")
    
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")
    
    # Check for generated model files
    print("\n📁 Generated model files:")
    expected_files = [
        ("models/streamlit_random_forest.pkl", "Random Forest model"),
        ("models/streamlit_scaler_rf.pkl", "Random Forest scaler"),
        ("models/streamlit_xgboost.pkl", "XGBoost model"),
        ("models/streamlit_scaler_xgb.pkl", "XGBoost scaler"),
        ("models/streamlit_ann.keras", "ANN model"),
        ("models/streamlit_scaler_ann.pkl", "ANN scaler"),
        ("models/streamlit_neuralfoil.pth", "NeuralFoil model"),
        ("models/streamlit_scaler_nf.pkl", "NeuralFoil scaler")
    ]
    
    all_files_present = True
    for file_path, description in expected_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  ✓ {description}: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"  ✗ {description}: {file_path} (not found)")
            all_files_present = False
    
    print("\n" + "=" * 70)
    
    if len(successful_models) == len(models_to_train) and all_files_present:
        print("🎉 All models trained successfully and files are present!")
        print("You can now use the Streamlit app to compare model performance.")
        return True
    else:
        print("⚠️  Some models failed or files are missing. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)