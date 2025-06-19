# Airfoil Aerodynamic Coefficient Regression

Machine learning models for predicting lift and drag coefficients of airfoils using the Stec8 database.

## Quick Start

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare training data
python scripts/prepare_data.py

# 3. Train all models  
python scripts/train_all_models.py

# 4. Compare model predictions
python scripts/compare_models.py

# To run the UI:
streamlit run app.py
```

## Project Structure

```
├── src/
│   ├── data_processing/     # Data extraction and preprocessing
│   ├── models/             # Model training scripts
│   └── analysis/           # Testing and visualization
├── scripts/                # Convenient run scripts
├── models/                 # Trained model files and scalers
├── Stec8/                  # Raw airfoil database
├── airfoil_tuner/          # Hyperparameter tuning results
├── xfoil-python/           # XFOIL Python wrapper
└── figs/                   # Generated plots
```

## Data

Training data extracted from Stec8 low-speed airfoil database containing:
- 69 x/y coordinate pairs per airfoil
- Reynolds number and angle of attack
- Measured lift and drag coefficients

## Individual Training

```bash
# Data preparation
cd src/data_processing && python make_csv.py

# Train specific models
cd src/models && python random_forest.py
cd src/models && python xgboost_model.py  
cd src/models && python ann.py
cd src/models && python neuralfoil_ann.py

# Analysis
cd src/analysis && python test_models.py
cd src/analysis && python xfoil_alone.py
```
