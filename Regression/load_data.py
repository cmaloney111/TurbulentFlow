import aerosandbox as asb
import aerosandbox.numpy as np
import polars as pl
from pathlib import Path
import sys, os
import ast
import pandas as pd
import csv
from neuralfoil import get_aero_from_airfoil

sys.path.append(str(Path(__file__).parent))

from neuralfoil.gen2_architecture._basic_data_type import Data

cols = Data.get_vector_column_names()

data_directory = Path(__file__).parent

raw_dfs = {}


csv_file = 'test.csv'

def extract_coordinates(coord_string):
    coords = ast.literal_eval(coord_string)
    
    x_values = [x for x, y in coords]
    y_values = [y for x, y in coords]

    x_values = x_values + [0] * (69 - len(x_values))
    y_values = y_values + [0] * (69 - len(y_values))
    return x_values, y_values

df = pd.read_csv("training_data_stec8.csv")
df[['x_coords', 'y_coords']] = df['coordinates'].apply(lambda x: pd.Series(extract_coordinates(x)))

airfoils = df['airfoil_name'].unique()

reynolds_to_angles = {
    airfoil_name: df[df['airfoil_name'] == airfoil_name]['reynolds_number'].tolist()
    for airfoil_name in airfoils
}

all_rows = []
for airfoil_name in airfoils:
    airfoil_data = df[df['airfoil_name'] == airfoil_name]
    reynolds_numbers = airfoil_data['reynolds_number'].unique()
    reynolds_to_angles = {
        int(reynolds): airfoil_data[airfoil_data['reynolds_number'] == reynolds]['angle_of_attack'].tolist()
        for reynolds in reynolds_numbers
    }
        
    x_coords_df = pd.DataFrame(df['x_coords'].tolist(), index=df.index)
    y_coords_df = pd.DataFrame(df['y_coords'].tolist(), index=df.index)


    x_numpy = x_coords_df.iloc[0].to_numpy()
    y_numpy = y_coords_df.iloc[0].to_numpy()
    coords_numpy = np.stack((x_numpy, y_numpy), axis=-1)
    coords_numpy = np.array([row for row in coords_numpy if not np.all(row == 0)])
    airfoil = asb.Airfoil(name=airfoil_name, coordinates=coords_numpy).to_kulfan_airfoil()

    normalization_outputs = airfoil.normalize(return_dict=True)
    normalized_airfoil = normalization_outputs["airfoil"].to_kulfan_airfoil(
        n_weights_per_side=8,
        normalize_coordinates=False 
    )
    print(normalized_airfoil.kulfan_parameters)
    # import os
    # os._exit(0)

    # preds = get_aero_from_airfoil(airfoil, np.array(reynolds_to_angles[reynolds]), reynolds, model_size='xxxlarge')

    row_values = list(normalized_airfoil.kulfan_parameters['upper_weights']) + list(normalized_airfoil.kulfan_parameters['lower_weights']) + [normalized_airfoil.kulfan_parameters['leading_edge_weight'], normalized_airfoil.kulfan_parameters['TE_thickness']]
    for reynolds in reynolds_numbers:
        for alpha in reynolds_to_angles:
            CL = airfoil_data.loc[
                (airfoil_data['reynolds_number'] == reynolds) & 
                (airfoil_data['angle_of_attack'] == alpha), 
                'lift_coefficient'
            ]
            CD = airfoil_data.loc[
                (airfoil_data['reynolds_number'] == reynolds) & 
                (airfoil_data['angle_of_attack'] == alpha), 
                'drag_coefficient'
            ]
            zeros_list = [0] * 32 * 6
            row_values.append(alpha)
            row_values.append(reynolds)
            row_values.extend([0, 9, 1, 1, 0.1, CL, CD, 0, 0, 0])
            row_values.extend(zeros_list)
            all_rows.append(row_values)


with open('test.csv', mode='w') as file:
    for row in all_rows:
        writer = csv.writer(file)
        writer.writerow(row)

import os
os._exit(0)
    
raw_dfs[csv_file.stem] = pl.read_csv(
    csv_file, has_header=False,
    dtypes={
        col: pl.Float32
        for col in cols
    }
)
print(f"\t{len(raw_dfs[csv_file.stem])} rows")

df = pl.concat(raw_dfs.values())

# Do some basic cleanup
cols_to_nullify = Data.get_vector_output_column_names().copy()
cols_to_nullify.remove("analysis_confidence")

c = pl.col("CD") <= 0
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with CD <= 0..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal([
                          pl.col(f"upper_bl_theta_{i}") <= 0
                          for i in range(Data.N)
                      ] + [
                          pl.col(f"lower_bl_theta_{i}") <= 0
                          for i in range(Data.N)
                      ])
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with nonpositive boundary layer thetas..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal([
                          pl.col(f"upper_bl_H_{i}") < 1
                          for i in range(Data.N)
                      ] + [
                          pl.col(f"lower_bl_H_{i}") < 1
                          for i in range(Data.N)
                      ])
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with H < 1 (non-physical BL)..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(sum([
    [
        pl.col(f"upper_bl_ue/vinf_{i}") < -20,
        pl.col(f"upper_bl_ue/vinf_{i}") > 20,
        pl.col(f"lower_bl_ue/vinf_{i}") < -20,
        pl.col(f"lower_bl_ue/vinf_{i}") > 20,
    ]
    for i in range(Data.N)
], start=[])
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with non-physical edge velocities..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    pl.col("Top_Xtr") < 0,
    pl.col("Top_Xtr") > 1,
    pl.col("Bot_Xtr") < 0,
    pl.col("Bot_Xtr") > 1,
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with non-physical transition locations..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

print("Dataset:")
print(df)
# print("Dataset statistics:")
# print(df.describe())

### Shuffle the training set (deterministically)
df = df.sample(
    fraction=1,
    with_replacement=False,
    shuffle=True,
    seed=0
)

# Make the scaled datasets
df_inputs_scaled = pl.DataFrame({
    **{
        f"s_kulfan_upper_{i}": df[f"kulfan_upper_{i}"]
        for i in range(8)
    },
    **{
        f"s_kulfan_lower_{i}": df[f"kulfan_lower_{i}"]
        for i in range(8)
    },
    "s_kulfan_LE_weight"   : df["kulfan_LE_weight"],
    "s_kulfan_TE_thickness": df["kulfan_TE_thickness"] * 50,
    "s_sin_2a"             : np.sind(2 * df["alpha"]),
    "s_cos_a"              : np.cosd(df["alpha"]),
    "s_1mcos2_a"           : 1 - np.cosd(df["alpha"]) ** 2,
    "s_Re"                 : (np.log(df["Re"]) - 12.5) / 3.5,
    "s_n_crit"             : (df["n_crit"] - 9) / 4.5,
    "s_xtr_upper"          : df["xtr_upper"],
    "s_xtr_lower"          : df["xtr_lower"],
})

di = df_inputs_scaled.describe()

df_outputs_scaled = pl.DataFrame({
    "s_analysis_confidence": df["analysis_confidence"],
    "s_CL"                 : 2 * df["CL"],
    "s_ln_CD"              : np.log(df["CD"]) / 2 + 2,
    "s_CM"                 : 20 * df["CM"],
    "s_Top_Xtr"            : df["Top_Xtr"],
    "s_Bot_Xtr"            : df["Bot_Xtr"],
    **{
        f"s_upper_bl_ret_{i}": np.log10(np.abs(df[f"upper_bl_ue/vinf_{i}"]) * df[f"upper_bl_theta_{i}"] * df["Re"] + 0.1)
        for i in range(Data.N)
    },
    **{
        f"s_upper_bl_H_{i}": np.log(df[f"upper_bl_H_{i}"] / 2.6)
        for i in range(Data.N)
    },
    **{
        f"s_upper_bl_ue/vinf_{i}": df[f"upper_bl_ue/vinf_{i}"]
        for i in range(Data.N)
    },
    **{
        f"s_lower_bl_ret_{i}": np.log10(np.abs(df[f"lower_bl_ue/vinf_{i}"]) * df[f"lower_bl_theta_{i}"] * df["Re"] + 0.1)
        for i in range(Data.N)
    },
    **{
        f"s_lower_bl_H_{i}": np.log(df[f"lower_bl_H_{i}"] / 2.6)
        for i in range(Data.N)
    },
    **{
        f"s_lower_bl_ue/vinf_{i}": df[f"lower_bl_ue/vinf_{i}"]
        for i in range(Data.N)
    },
})

do = df_outputs_scaled.describe([0.01, 0.99])

### Split the dataset into train and test sets
test_train_split_index = int(len(df) * 0.95)
# df_train = df[:test_train_split_index]
# df_test = df[test_train_split_index:]
df_train_inputs_scaled = df_inputs_scaled[:test_train_split_index]
df_train_outputs_scaled = df_outputs_scaled[:test_train_split_index]
df_test_inputs_scaled = df_inputs_scaled[test_train_split_index:]
df_test_outputs_scaled = df_outputs_scaled[test_train_split_index:]

mean_inputs_scaled = np.mean(df_inputs_scaled.to_numpy(), axis=0)
cov_inputs_scaled = np.cov(df_inputs_scaled.to_numpy(), rowvar=False)


def make_data(row_index, df=df):
    row = df[row_index]
    return Data.from_vector(
        row[cols].to_numpy().flatten()
    )


if __name__ == '__main__':
    d = make_data(len(df) // 2)
