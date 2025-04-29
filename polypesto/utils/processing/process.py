import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Optional

# Function(s) to merge simple csv data into one dataframe containing Time, Conversion A, Conversion B

def process_data(monomer_A: str, monomer_B: str):
    df_A = pd.read_csv(monomer_A)
    df_B = pd.read_csv(monomer_B)

    df_A_interp = interpolate_data(df_A, df_B)

    df_A_smooth = smooth_data(df_A_interp)
    df_B_smooth = smooth_data(df_B)

    df_A_downsample = downsample_data(df_A_smooth)
    df_B_downsample = downsample_data(df_B_smooth)

    df_merged = merge_data(df_A_downsample, df_B_downsample)
    
    return df_merged

def downsample_data(df: pd.DataFrame) -> pd.DataFrame:
    df_down = df.iloc[::3].reset_index(drop=True)
    return df_down

def smooth_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Conversion [%]'] = gaussian_filter1d(df['Conversion [%]'], sigma=2)

    return df

# def remove_noise(df: pd.DataFrame, window: int) -> pd.DataFrame:
#     conv = df['Conversion [%]'].values
#     diffs = np.diff(conv)

#     start_idx = 0
#     for i in range(len(diffs) - window):
#         if np.all(diffs[i:i+window] > 0):
#             start_idx = i + 1  
#             break

#     df_clean = df.copy()
#     df_clean.loc[:start_idx, 'Conversion [%]'] = np.nan

#     return df_clean.reset_index(drop=True)

def interpolate_data(df_A: pd.DataFrame, df_B: pd.DataFrame) -> pd.DataFrame:
   conversion_interp = (
    df_A.set_index('Time [min]')['Conversion [%]']
        .reindex(df_B['Time [min]'])
        .interpolate(method='linear')
        .reset_index()
    )
   
   df_A_interp = pd.DataFrame({
        'Time [min]': conversion_interp['Time [min]'],
        'Conversion [%]': conversion_interp['Conversion [%]']
    })
   
   return df_A_interp

def merge_data(df_A: pd.DataFrame, df_B: pd.DataFrame) -> pd.DataFrame:
    df_A_renamed = df_A.rename(columns={'Conversion [%]': 'Conversion A [%]'})
    df_B_renamed = df_B.rename(columns={'Conversion [%]': 'Conversion B [%]'})

    df_merged = pd.merge(df_A_renamed, df_B_renamed, on='Time [min]', how='inner')

    return df_merged

# Function(s) to convert Time vs. Conversion to Total Conversion vs. Monomer Conversion

def convert_axes(df: pd.DataFrame, fA0: float) -> pd.DataFrame:
    df.rename(columns={'Time [min]': 'Total Conversion [%]'}, inplace=True)
    df['Total Conversion [%]'] = fA0 * df['Conversion A [%]'] + (1 - fA0) * df['Conversion B [%]']

    return df

# Function(s) to plot dataframes

def plot_orig_df(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.plot(df['Time [min]'], df['Conversion A [%]'], label='Conversion A', color='blue')
    plt.plot(df['Time [min]'], df['Conversion B [%]'], label='Conversion B', color='green')
    plt.xlabel('Time [min]')
    plt.ylabel('Monomer Conversion [%]')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title('Time vs Monomer Conversion')
    plt.show()

def plot_conv_df(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.plot(df['Total Conversion [%]'], df['Conversion A [%]'], label='Conversion A', color='blue')
    plt.plot(df['Total Conversion [%]'], df['Conversion B [%]'], label='Conversion B', color='green')
    plt.xlabel('Total Conversion [%]')
    plt.ylabel('Monomer Conversion [%]')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title('Monomer Conversion vs Total Conversion')
    plt.show()

# Function(s) to convert dataframe into petab format

def convert_petab(df: pd.DataFrame, condition_id: str) -> pd.DataFrame:
    df_conv_A = df.copy()
    df_conv_A.drop(columns=['Conversion B [%]'], inplace=True)
    df_conv_A.rename(columns={'Total Conversion [%]': 'time', 'Conversion A [%]': 'measurement'}, inplace=True)
    df_conv_A['simulationConditionId'] = condition_id
    df_conv_A['observableId'] = 'obs_xA'

    df_conv_B = df.copy()
    df_conv_B.drop(columns=['Conversion A [%]'], inplace=True)
    df_conv_B.rename(columns={'Total Conversion [%]': 'time', 'Conversion B [%]': 'measurement'}, inplace=True)
    df_conv_B['simulationConditionId'] = condition_id
    df_conv_B['observableId'] = 'obs_xB'

    df_conv = pd.concat([df_conv_A, df_conv_B], ignore_index=True)
    df_conv = df_conv[['observableId', 'simulationConditionId', 'time', 'measurement']]
    return df_conv

def save_tsv(df: pd.DataFrame, output_file: str):
    df.to_csv(output_file, sep='\t', index=False)

# Function(s) to create conditions dataframe

def create_conditions(df: pd.DataFrame, condition_id: str, fA0: float) -> pd.DataFrame:
    new_row = {'conditionId': condition_id, 'A0': fA0, 'B0': 1 - fA0, 'conditionName': condition_id}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index = True)
    return df

# Function(s) to join dataframes for measurements and conditions

def join_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df_joined = pd.concat(dfs, ignore_index=True)
    return df_joined