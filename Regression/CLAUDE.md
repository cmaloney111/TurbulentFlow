# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an aerodynamic coefficient regression project that predicts lift and drag coefficients for airfoils using machine learning models. The project uses the Stec8 airfoil database and implements multiple regression approaches including Random Forest, XGBoost, Artificial Neural Networks (ANN), and NeuralFoil-based neural networks.

## Key Data Files

- `training_data_stec8.csv`: Main training dataset extracted from Stec8 database with airfoil coordinates, Reynolds numbers, angles of attack, and aerodynamic coefficients
- `training_data_new.csv`: Processed version of the Stec8 data with expanded coordinate features
- `Stec8/ALL.PD`: Raw performance data from the Stec8 airfoil database
- `Stec8/*.COR`: Coordinate files for individual airfoils

## Quick Start Commands

### All-in-one Scripts
```bash
python scripts/prepare_data.py      # Extract data from Stec8/ALL.PD 
python scripts/train_all_models.py  # Train all models sequentially
python scripts/compare_models.py    # Interactive model comparison
```

### Individual Training
```bash
# Data preparation
cd src/data_processing && python make_csv.py

# Model training  
cd src/models && python random_forest.py     # → models/first_random_forest.pkl
cd src/models && python xgboost_model.py     # → models/first_xgboost.pkl
cd src/models && python ann.py               # → models/best_neural_network_model.keras
cd src/models && python neuralfoil_ann.py    # → models/neuralfoil-nn.pth

# Analysis and testing
cd src/analysis && python test_models.py     # Interactive model comparison
cd src/analysis && python xfoil_alone.py     # Standalone XFOIL analysis
cd src/analysis && python xfoil_comp.py      # XFOIL vs predictions
```

## Architecture

### Project Structure
```
src/
├── data_processing/     # Data extraction and preprocessing
├── models/             # Model training implementations  
└── analysis/           # Testing and visualization tools
scripts/                # Convenient run scripts
models/                 # Trained models and scalers
```

### Data Processing Pipeline
1. **Raw Data**: Stec8 database contains airfoil coordinates and performance measurements
2. **Preprocessing**: `src/data_processing/make_csv.py` extracts and structures data from `ALL.PD` format
3. **Feature Engineering**: Coordinate points are flattened into feature vectors (x_0, x_1, ..., y_0, y_1, ...)
4. **Scaling**: StandardScaler applied to inputs, scalers saved for prediction consistency

### Model Implementations
- **Random Forest**: `src/models/random_forest.py` - Scikit-learn RandomForestRegressor with hyperparameter tuning
- **XGBoost**: `src/models/xgboost_model.py` - XGBoost regressor with RandomizedSearchCV optimization  
- **Keras ANN**: `src/models/ann.py` - Deep neural network with dropout and early stopping
- **PyTorch Net**: `src/models/neuralfoil_ann.py` - Custom PyTorch network inspired by NeuralFoil architecture

### XFOIL Integration
The project includes a Python-wrapped XFOIL library (`xfoil-python/`) for aerodynamic analysis validation. XFOIL is a widely-used panel method for airfoil analysis.

## Important Notes

- Models expect 140-dimensional input vectors (69 x-coordinates + 69 y-coordinates + Reynolds number + angle of attack)
- All models predict 2 outputs: lift_coefficient and drag_coefficient  
- Scalers are saved alongside models and must be loaded for consistent predictions
- The `src/analysis/test_models.py` script provides interactive model comparison with visualization
- Coordinate data follows Eppler format: trailing edge → upper surface → leading edge → lower surface → trailing edge

## Dependencies

Key Python packages required:
- pandas, numpy, matplotlib
- scikit-learn  
- xgboost
- tensorflow/keras
- torch
- aerosandbox (for Kulfan parameterization)
- neuralfoil (for advanced airfoil analysis)
- Custom xfoil package (in xfoil-python/)