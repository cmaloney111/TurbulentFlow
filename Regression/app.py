import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ast
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import keras

# Configure page
st.set_page_config(
    page_title="Aerodynamic Model Trainer & Comparator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .airfoil-card {
        border: 2px solid #e1e4e8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .airfoil-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .airfoil-card.selected {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_airfoils' not in st.session_state:
    st.session_state.selected_airfoils = []
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []

def load_airfoil_coordinates():
    """Load airfoil coordinate files from Stec8 directory"""
    stec8_dir = Path("Stec8")
    airfoil_coords = {}
    
    if not stec8_dir.exists():
        st.error("Stec8 directory not found!")
        return {}
    
    for cor_file in stec8_dir.glob("*.COR"):
        airfoil_name = cor_file.stem
        try:
            with open(cor_file, 'r') as f:
                lines = f.readlines()
                coords = []
                # Skip the first line (airfoil name) and parse coordinates
                for line in lines[1:]:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                x, y = float(parts[0]), float(parts[1])
                                coords.append((x, y))
                            except ValueError:
                                continue
                if coords:
                    airfoil_coords[airfoil_name] = coords
        except Exception as e:
            st.warning(f"Could not read {cor_file}: {e}")
    
    return airfoil_coords

def plot_airfoil(coordinates, title="Airfoil"):
    """Create a plotly figure for an airfoil"""
    if not coordinates:
        return None
    
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='lines+markers',
        name=title,
        line=dict(color='white', width=2),
        marker=dict(size=3)
    ))
    
    fig.update_layout(
        title=title,
        width=300,
        height=200,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            range=[-0.1, 1.1],
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            title=""
        ),
        yaxis=dict(
            scaleanchor="x", 
            scaleratio=1,
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            title=""
        )
    )
    
    return fig

def get_available_reynolds_numbers(selected_airfoils=None):
    """Get available Reynolds numbers from training data, optionally filtered by selected airfoils"""
    try:
        df = pd.read_csv('training_data_stec8.csv')
        
        # Filter by selected airfoils if provided
        if selected_airfoils:
            df = df[df['airfoil_name'].isin(selected_airfoils)]
        
        reynolds_numbers = sorted(df['reynolds_number'].unique())
        return reynolds_numbers
    except:
        return [59400.0, 100000.0, 200000.0, 500000.0, 1000000.0]

def plot_matplotlib_airfoil_comparison(airfoil_name, selected_models, selected_reynolds):
    """Create matplotlib plot with predictions vs ground truth comparison"""
    try:
        from model_utils import load_model_for_prediction, predict_coefficients, extract_coordinates
        
        # Load training data
        df = pd.read_csv('training_data_stec8.csv')
        
        # Filter data by airfoil name
        airfoil_data = df[df['airfoil_name'] == airfoil_name]
        if airfoil_data.empty:
            st.error(f"No data found for airfoil '{airfoil_name}'")
            return None
        
        # Get airfoil shape and coordinates for inset
        shape_str = airfoil_data.iloc[0]['coordinates']
        airfoil_shape = np.array(ast.literal_eval(shape_str))
        
        # Filter by selected Reynolds numbers
        reynolds_data = airfoil_data[airfoil_data['reynolds_number'].isin(selected_reynolds)]
        reynolds_numbers = reynolds_data['reynolds_number'].unique()
        
        if len(reynolds_numbers) == 0:
            st.warning(f"No data found for selected Reynolds numbers for {airfoil_name}")
            return None
        
        # Color map for Reynolds numbers
        cmap = cm.get_cmap('tab10', len(reynolds_numbers))
        
        # Color map for models - different colors for each model
        model_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # orange, green, red, purple
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(f"Model Predictions vs Ground Truth: {airfoil_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Drag Coefficient (Cd)", fontsize=12)
        ax.set_ylabel("Lift Coefficient (Cl)", fontsize=12)
        
        # Model mapping and configuration
        model_mapping = {
            "Random Forest": ("random_forest", "RF"),
            "XGBoost": ("xgboost", "XGB"), 
            "ANN (Neural Network)": ("ann", "ANN"),
            "NeuralFoil": ("neuralfoil", "NF")
        }
        
        # Load available models
        loaded_models = {}
        for model_name in selected_models:
            if model_name in model_mapping:
                model_key, short_name = model_mapping[model_name]
                try:
                    model, scaler = load_model_for_prediction(model_key)
                    loaded_models[model_name] = (model, scaler, model_key, short_name)
                except Exception as e:
                    st.warning(f"Could not load {model_name}: {e}")
                    continue
        
        # Plot for each Reynolds number
        for i, reynolds in enumerate(reynolds_numbers):
            reynolds_subset = reynolds_data[reynolds_data['reynolds_number'] == reynolds]
            
            # Remove duplicates and sort
            reynolds_subset = reynolds_subset.drop_duplicates(subset='drag_coefficient')
            reynolds_subset = reynolds_subset.sort_values(by='angle_of_attack')
            
            x_gt = reynolds_subset['drag_coefficient']
            y_gt = reynolds_subset['lift_coefficient']
            
            color = cmap(i)
            
            # Plot ground truth data with distinct styling
            ax.plot(x_gt, y_gt, linestyle='-', linewidth=3, alpha=0.8, 
                   label=f"Ground Truth (Re={reynolds:.0e})", color=color, zorder=10)
            ax.scatter(x_gt, y_gt, s=60, color=color, alpha=0.8, 
                      marker='o', edgecolors='black', linewidth=1, zorder=11)
            
            # Generate predictions for each model
            if loaded_models:
                # Extract coordinates and features for prediction
                reynolds_subset_copy = reynolds_subset.copy()
                reynolds_subset_copy[['x_coords', 'y_coords']] = reynolds_subset_copy['coordinates'].apply(
                    lambda x: pd.Series(extract_coordinates(x))
                )
                
                # Prepare feature matrix
                x_coords_df = pd.DataFrame(reynolds_subset_copy['x_coords'].tolist(), index=reynolds_subset_copy.index)
                y_coords_df = pd.DataFrame(reynolds_subset_copy['y_coords'].tolist(), index=reynolds_subset_copy.index)
                x_coords_df.columns = [f'x_{j}' for j in range(x_coords_df.shape[1])]
                y_coords_df.columns = [f'y_{j}' for j in range(y_coords_df.shape[1])]
                
                features_df = pd.concat([
                    reynolds_subset_copy[['reynolds_number', 'angle_of_attack']].reset_index(drop=True),
                    x_coords_df.reset_index(drop=True), 
                    y_coords_df.reset_index(drop=True)
                ], axis=1)
                
                # Plot predictions for each model
                for j, (model_name, (model, scaler, model_key, short_name)) in enumerate(loaded_models.items()):
                    try:
                        # Make predictions
                        features_scaled = scaler.transform(features_df)
                        
                        if model_key == 'ann':
                            predictions = model.predict(features_scaled)
                        elif model_key in ['random_forest', 'xgboost']:
                            predictions = model.predict(features_scaled)
                        elif model_key == 'neuralfoil':
                            import torch
                            predictions = model(torch.tensor(features_scaled, dtype=torch.float32))
                            predictions = predictions.detach().numpy()
                        
                        pred_cl = predictions[:, 0]
                        pred_cd = predictions[:, 1]
                        
                        # Use different colors for different models
                        model_color = model_colors[j % len(model_colors)]
                        
                        # Use different line styles for different models
                        line_styles = [':', '-.', '--', (0, (3, 1, 1, 1))]  # dotted, dashdot, dashed, custom
                        line_style = line_styles[j % len(line_styles)]
                        
                        # Plot predictions with distinct styling - use model color instead of Reynolds color
                        ax.plot(pred_cd, pred_cl, linestyle=line_style, linewidth=2.5, 
                               label=f"{short_name} Prediction (Re={reynolds:.0e})", 
                               color=model_color, alpha=0.9, zorder=8)
                        ax.scatter(pred_cd, pred_cl, s=40, color=model_color, alpha=0.7, 
                                  marker='s', edgecolors='white', linewidth=0.5, zorder=9)
                        
                    except Exception as e:
                        st.warning(f"Error generating predictions for {model_name}: {e}")
        
        # Enhance legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
                 fancybox=True, shadow=True, fontsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add inset for airfoil shape
        inset_ax = fig.add_axes([0.02, 0.75, 0.25, 0.2])
        inset_ax.plot(airfoil_shape[:, 0], airfoil_shape[:, 1], color='black', linewidth=2)
        inset_ax.fill(airfoil_shape[:, 0], airfoil_shape[:, 1], color='lightgray', alpha=0.7)
        inset_ax.set_title(f"{airfoil_name} Shape", fontsize=10, fontweight='bold')
        inset_ax.axis("equal")
        inset_ax.set_xlim(0, 1)
        inset_ax.axis("off")
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating matplotlib plot: {e}")
        return None

def main():
    st.markdown('<h1 class="main-header">✈️ Aerodynamic Model Trainer & Comparator</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Train Models", "Compare Models"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Train Models":
        show_training_page()
    elif page == "Compare Models":
        show_comparison_page()

def show_home_page():
    st.header("Welcome to the Aerodynamic Model Trainer & Comparator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Train Models")
        st.write("""
        Train machine learning models to predict aerodynamic coefficients:
        - **ANN (Neural Network)**: Deep learning approach with customizable architecture
        - **Random Forest**: Ensemble method with tunable parameters
        - **XGBoost**: Gradient boosting with hyperparameter optimization
        - **NeuralFoil**: PyTorch-based neural network
        """)
        
    with col2:
        st.subheader("Compare Models")
        st.write("""
        Compare model predictions across different airfoils:
        - Visual airfoil selection with coordinate plots
        - Reynolds number and angle of attack customization
        - Interactive performance comparison charts
        - Side-by-side coefficient predictions
        """)
    
    
    st.header("Project Overview")
    st.write("""
    This application provides an interface for training and comparing machine learning models 
    that predict aerodynamic coefficients (lift and drag) for various airfoils using the Stec8 database.
    
    **Key Features:**
    - Interactive airfoil visualization and selection
    - Customizable model training parameters
    - Real-time model comparison
    - Reynolds number sensitivity analysis
    """)

def show_training_page():
    st.header("🎯 Model Training")
    
    # Model selection
    st.subheader("Select Models to Train")
    model_options = {
        "ANN (Neural Network)": "ann_2",
        "Random Forest": "random_forest", 
        "XGBoost": "xgboost",
        "NeuralFoil": "neuralfoil"
    }
    
    selected_models = st.multiselect(
        "Choose models to train:",
        list(model_options.keys()),
        default=["Random Forest"]
    )
    
    if not selected_models:
        st.warning("Please select at least one model to train.")
        return
    
    # Training parameters section
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**General Parameters**")
        test_size = st.slider("Test set size (%)", 10, 30, 20) / 100
        random_state = st.number_input("Random state", 0, 1000, 42)
        
    with col2:
        st.write("**Data Parameters**")
        use_scaling = st.checkbox("Use feature scaling", True)
        remove_outliers = st.checkbox("Remove outliers", False)
    
    # Model-specific parameters
    for model_name in selected_models:
        model_key = model_options[model_name]
        st.subheader(f"{model_name} Parameters")
        
        if model_key == "ann_2":
            col1, col2, col3 = st.columns(3)
            with col1:
                ann_2_epochs = st.number_input(f"Epochs", 50, 1000, 500, key=f"{model_key}_epochs")
                ann_2_batch_size = st.selectbox(f"Batch Size", [1, 8, 16, 32, 64], index=0, key=f"{model_key}_batch")
            with col2:
                ann_2_learning_rate = st.selectbox(f"Learning Rate", [0.001, 0.01, 0.1], index=1, key=f"{model_key}_lr")
                ann_2_layers = st.text_input(f"Layer Sizes (comma-separated)", "2048,2048,2048,1024,512,256,128", key=f"{model_key}_layers")
            with col3:
                ann_2_activation = st.selectbox(f"Activation", ["relu", "tanh", "sigmoid"], key=f"{model_key}_activation")
                ann_2_optimizer = st.selectbox(f"Optimizer", ["Adam", "SGD", "RMSprop"], key=f"{model_key}_optimizer")
        
        elif model_key == "random_forest":
            col1, col2, col3 = st.columns(3)
            with col1:
                random_forest_n_estimators = st.number_input(f"Number of Trees", 100, 10000, 5000, key=f"{model_key}_trees")
                random_forest_max_depth = st.selectbox(f"Max Depth", [None, 10, 20, 30], key=f"{model_key}_depth")
            with col2:
                random_forest_min_samples_split = st.number_input(f"Min Samples Split", 2, 20, 2, key=f"{model_key}_split")
                random_forest_min_samples_leaf = st.number_input(f"Min Samples Leaf", 1, 10, 1, key=f"{model_key}_leaf")
            with col3:
                random_forest_max_features = st.selectbox(f"Max Features", [None, "sqrt", "log2"], key=f"{model_key}_features")
        
        elif model_key == "xgboost":
            col1, col2, col3 = st.columns(3)
            with col1:
                xgboost_n_estimators = st.number_input(f"Number of Estimators", 100, 5000, 2000, key=f"{model_key}_est")
                xgboost_max_depth = st.number_input(f"Max Depth", 3, 30, 15, key=f"{model_key}_depth")
            with col2:
                xgboost_learning_rate = st.selectbox(f"Learning Rate", [0.01, 0.05, 0.1, 0.2], index=1, key=f"{model_key}_lr")
                xgboost_subsample = st.slider(f"Subsample", 0.7, 1.0, 1.0, key=f"{model_key}_subsample")
            with col3:
                xgboost_colsample_bytree = st.slider(f"Column Sample", 0.7, 1.0, 1.0, key=f"{model_key}_colsample")
                xgboost_gamma = st.number_input(f"Gamma", 0.0, 1.0, 0.0, key=f"{model_key}_gamma")
        
        elif model_key == "neuralfoil":
            col1, col2 = st.columns(2)
            with col1:
                neuralfoil_epochs = st.number_input(f"Epochs", 100, 1000, 500, key=f"{model_key}_epochs")
                neuralfoil_batch_size = st.selectbox(f"Batch Size", [4, 8, 16, 32], index=1, key=f"{model_key}_batch")
            with col2:
                neuralfoil_learning_rate = st.selectbox(f"Learning Rate", [1e-5, 1e-4, 1e-3], index=0, key=f"{model_key}_lr")
                neuralfoil_hidden_layers = st.number_input(f"Hidden Layers", 2, 8, 4, key=f"{model_key}_layers")
                neuralfoil_width = st.number_input(f"Layer Width", 64, 512, 128, key=f"{model_key}_width")
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take several minutes."):
            train_models(selected_models, model_options, locals())

def train_models(selected_models, model_options, params_dict):
    """Train the selected models with given parameters"""
    from model_utils import (
        load_training_data, prepare_data, train_random_forest_model,
        train_xgboost_model, train_ann_model, train_neuralfoil_model
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load and prepare data
    status_text.text("Loading training data...")
    try:
        data = load_training_data()
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            data, 
            test_size=params_dict.get('test_size', 0.2),
            random_state=params_dict.get('random_state', 42),
            use_scaling=params_dict.get('use_scaling', True)
        )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    results = {}
    
    for i, model_name in enumerate(selected_models):
        model_key = model_options[model_name]
        status_text.text(f"Training {model_name}...")
        
        try:
            # Extract model-specific parameters
            model_params = {k: v for k, v in params_dict.items() if k.startswith(f"{model_key}_")}
            model_params = {k.replace(f"{model_key}_", ""): v for k, v in model_params.items()}
            model_params['random_state'] = params_dict.get('random_state', 42)
            
            # Train the appropriate model
            if model_key == "random_forest":
                model, metrics = train_random_forest_model(
                    X_train, y_train, X_test, y_test, scaler, **model_params
                )
            elif model_key == "xgboost":
                model, metrics = train_xgboost_model(
                    X_train, y_train, X_test, y_test, scaler, **model_params
                )
            elif model_key == "ann_2":
                model, metrics = train_ann_model(
                    X_train, y_train, X_test, y_test, scaler, **model_params
                )
            elif model_key == "neuralfoil":
                model, metrics = train_neuralfoil_model(
                    X_train, y_train, X_test, y_test, scaler, **model_params
                )
            
            results[model_name] = metrics
            progress_bar.progress((i + 1) / len(selected_models))
        
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    status_text.text("Training complete!")
    
    # Display results
    st.subheader("Training Results")
    
    # Create a clean results dataframe
    clean_results = {}
    for model, metrics in results.items():
        if "error" not in metrics:
            clean_results[model] = {
                "MAE": f"{metrics['mae']:.4f}",
                "MSE": f"{metrics['mse']:.4f}",
                "R² Score": f"{metrics['r2']:.4f}"
            }
        else:
            clean_results[model] = {"Error": metrics["error"]}
    
    results_df = pd.DataFrame(clean_results).T
    st.dataframe(results_df, use_container_width=True)
    
    # Plot results for successful models
    successful_models = {k: v for k, v in results.items() if "error" not in v}
    if len(successful_models) > 1:
        fig = px.bar(
            x=list(successful_models.keys()),
            y=[successful_models[k]['r2'] for k in successful_models.keys()],
            title="Model R² Scores Comparison",
            labels={'x': 'Model', 'y': 'R² Score'},
            color=[successful_models[k]['r2'] for k in successful_models.keys()],
            color_continuous_scale="viridis"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_comparison_page():
    st.header("📊 Model Comparison")
    
    # Load airfoil coordinates
    airfoil_coords = load_airfoil_coordinates()
    
    if not airfoil_coords:
        st.error("No airfoil coordinate files found in Stec8 directory!")
        return
    
    # Airfoil selection
    st.subheader("Select Airfoils")
    
    # Create airfoil selection grid
    cols_per_row = 4
    airfoil_names = list(airfoil_coords.keys())
    
    # Create selection interface
    with st.container():
        for i in range(0, len(airfoil_names), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(airfoil_names):
                    airfoil_name = airfoil_names[i + j]
                    with col:
                        # Create airfoil plot
                        fig = plot_airfoil(airfoil_coords[airfoil_name], airfoil_name)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Selection checkbox
                        is_selected = st.checkbox(
                            f"Select {airfoil_name}", 
                            key=f"airfoil_{airfoil_name}",
                            value=airfoil_name in st.session_state.selected_airfoils
                        )
                        
                        # Update session state
                        if is_selected and airfoil_name not in st.session_state.selected_airfoils:
                            st.session_state.selected_airfoils.append(airfoil_name)
                        elif not is_selected and airfoil_name in st.session_state.selected_airfoils:
                            st.session_state.selected_airfoils.remove(airfoil_name)
    
    # Model and parameter selection
    st.subheader("Comparison Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Models to Compare**")
        available_models = ["Random Forest", "XGBoost", "ANN (Neural Network)", "NeuralFoil"]
        selected_models = st.multiselect(
            "Select models:",
            available_models,
            default=["Random Forest", "XGBoost"]
        )
    
    with col2:
        st.write("**Reynolds Numbers**")
        reynolds_numbers = get_available_reynolds_numbers(st.session_state.selected_airfoils)
        selected_reynolds = st.multiselect(
            "Select Reynolds numbers:",
            reynolds_numbers,
            default=[reynolds_numbers[0]] if reynolds_numbers else []
        )
    
    with col3:
        st.write("**Angle of Attack Range**")
        angle_min = st.number_input("Min angle (°)", -10.0, 0.0, -5.0)
        angle_max = st.number_input("Max angle (°)", 0.0, 20.0, 15.0)
        angle_step = st.number_input("Step size (°)", 0.5, 5.0, 1.0)
    
    # Generate comparison
    if st.button("🔄 Generate Comparison", type="primary"):
        if not st.session_state.selected_airfoils:
            st.warning("Please select at least one airfoil.")
        elif not selected_models:
            st.warning("Please select at least one model.")
        elif not selected_reynolds:
            st.warning("Please select at least one Reynolds number.")
        else:
            generate_comparison(
                st.session_state.selected_airfoils,
                selected_models,
                selected_reynolds,
                angle_min,
                angle_max,
                angle_step,
                airfoil_coords
            )

def generate_comparison(airfoils, models, reynolds_numbers, angle_min, angle_max, angle_step, airfoil_coords):
    """Generate comparison plots and data"""
    from model_utils import load_model_for_prediction, predict_coefficients
    
    st.subheader("Comparison Results")
    
    # Create angle of attack array
    angles = np.arange(angle_min, angle_max + angle_step, angle_step)
    
    # Model mapping
    model_mapping = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost", 
        "ANN (Neural Network)": "ann",
        "NeuralFoil": "neuralfoil"
    }
    
    # Load available models
    loaded_models = {}
    for model_name in models:
        model_key = model_mapping.get(model_name)
        if model_key:
            try:
                model, scaler = load_model_for_prediction(model_key)
                loaded_models[model_name] = (model, scaler, model_key)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
    
    if not loaded_models:
        st.error("No trained models found! Please train models first on the Training page.")
        return
    
    # Create comparison plots
    for airfoil in airfoils:
        st.write(f"### {airfoil}")
        
        if airfoil not in airfoil_coords:
            st.error(f"Coordinates not found for {airfoil}")
            continue
        
        # Create subplots for lift and drag coefficients
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Lift Coefficient', 'Drag Coefficient'),
            shared_xaxes=True
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (model_name, (model, scaler, model_key)) in enumerate(loaded_models.items()):
            for j, reynolds in enumerate(reynolds_numbers):
                try:
                    # Make predictions for all angles
                    cl_data = []
                    cd_data = [] 
                    
                    for angle in angles:
                        prediction = predict_coefficients(
                            model, scaler, airfoil_coords[airfoil], 
                            reynolds, angle, model_key
                        )
                        cl_data.append(prediction[0])  # lift coefficient
                        cd_data.append(prediction[1])  # drag coefficient
                    
                    # Add traces
                    color = colors[i % len(colors)]
                    line_style = dict(color=color)
                    if len(reynolds_numbers) > 1:
                        line_style['dash'] = 'dash' if j == 1 else 'solid'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=angles, y=cl_data,
                            name=f"{model_name} (Re={reynolds:.0e})",
                            line=line_style,
                            legendgroup=f"{model_name}_{reynolds}"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=angles, y=cd_data,
                            name=f"{model_name} (Re={reynolds:.0e})",
                            line=line_style,
                            showlegend=False,
                            legendgroup=f"{model_name}_{reynolds}"
                        ),
                        row=1, col=2
                    )
                
                except Exception as e:
                    st.warning(f"Error making predictions with {model_name}: {e}")
        
        fig.update_xaxes(title_text="Angle of Attack (°)")
        fig.update_yaxes(title_text="Cl", row=1, col=1)
        fig.update_yaxes(title_text="Cd", row=1, col=2)
        fig.update_layout(height=400, title=f"{airfoil} Performance Comparison")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add matplotlib plot with predictions vs ground truth
        st.subheader(f"Predictions vs Ground Truth: {airfoil}")
        matplotlib_fig = plot_matplotlib_airfoil_comparison(airfoil, models, reynolds_numbers)
        if matplotlib_fig:
            st.pyplot(matplotlib_fig)
            plt.close(matplotlib_fig)  # Close to free memory
        
        # Show airfoil shape
        with st.expander(f"View {airfoil} Shape"):
            airfoil_fig = plot_airfoil(airfoil_coords[airfoil], airfoil)
            if airfoil_fig:
                airfoil_fig.update_layout(height=300)
                st.plotly_chart(airfoil_fig, use_container_width=True)
        
        # Show prediction table
        with st.expander(f"View {airfoil} Prediction Data"):
            prediction_data = []
            for model_name, (model, scaler, model_key) in loaded_models.items():
                for reynolds in reynolds_numbers:
                    for angle in angles[::2]:  # Show every other angle to keep table manageable
                        try:
                            prediction = predict_coefficients(
                                model, scaler, airfoil_coords[airfoil], 
                                reynolds, angle, model_key
                            )
                            prediction_data.append({
                                'Model': model_name,
                                'Reynolds': f"{reynolds:.0e}",
                                'Angle (°)': f"{angle:.1f}",
                                'Cl': f"{prediction[0]:.4f}",
                                'Cd': f"{prediction[1]:.4f}"
                            })
                        except:
                            continue
            
            if prediction_data:
                pred_df = pd.DataFrame(prediction_data)
                st.dataframe(pred_df, use_container_width=True)

def create_prediction_vs_truth_plot(airfoil_name, models=['Random Forest', 'XGBoost'], reynolds_numbers=None):
    """
    Standalone function to create a clear prediction vs ground truth plot
    
    Args:
        airfoil_name: Name of the airfoil to plot
        models: List of model names to include in comparison
        reynolds_numbers: List of Reynolds numbers to include (if None, uses all available)
    
    Returns:
        matplotlib figure object
    """
    fig = plot_matplotlib_airfoil_comparison(airfoil_name, models, reynolds_numbers)
    return fig

if __name__ == "__main__":
    main()