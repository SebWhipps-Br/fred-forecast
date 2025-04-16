from pathlib import Path

import numpy as np
import pandas as pd

# Define the directory paths
BASE_DIR = Path('EA-MD-HT')
OUTPUT_DIR = Path('processed_data')
OUTPUT_DIR.mkdir(exist_ok=True)

'''
We take 37 macroeconomic indicators that have been collected at a monthly frequency for 8 countries 
(Austria, Belgium, Germany, Greece, Spain, France, Italy and the Netherlands), forming a matrix-valued 
(K = 2) time series of dimensions (p1, p2) = (8, 37); the data span the period from 2002-02 to 2023-09 (n = 257).
'''


def get_all_data():
    """Gets data from all xlsx files in directory"""
    data_dict = {}
    files = [f for f in BASE_DIR.glob('*dataM_HT.xlsx')]

    for file in files:
        country_code = file.name[:2]
        try:
            df = pd.read_excel(file, engine='openpyxl')
            data_dict[country_code] = df
            print(f"Loaded {country_code} data: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

    return data_dict


def check_datasets(data_dict):
    """Checks datasets balance and properties"""
    print("\nDataset Summary:")
    for country, df in data_dict.items():
        print(f"{country}: {df.shape[0]} rows, {df.shape[1]} columns")

    min_length = min(df.shape[0] for df in data_dict.values())
    print(f"Minimum time length across datasets: {min_length}")

    date_col = 'Time'
    print("\nDate range per country:")
    for country, df in data_dict.items():
        print(f"{country}: {df[date_col].min()} to {df[date_col].max()}")

    return date_col, min_length


def reshape_datasets(data_dict, date_col, min_length):
    """Reshapes datasets for consistency"""
    processed_dict = {}

    common_dates = set.intersection(*[set(df[date_col]) for df in data_dict.values()])
    common_dates = sorted(list(common_dates))[:min_length]
    for country, df in data_dict.items():
        processed_dict[country] = df[df[date_col].isin(common_dates)].sort_values(date_col)

    return processed_dict


def find_common_variables(data_dict):
    """Removes variables not common across all datasets, handling country postfixes"""

    def strip_country_code(col_name, country):
        if col_name.endswith(f"_{country}"):
            return col_name[:-len(f"_{country}")]
        return col_name

    base_vars_per_country = {}
    for country, df in data_dict.items():
        base_vars = {strip_country_code(col, country) for col in df.columns}
        base_vars_per_country[country] = base_vars

    common_base_vars = set.intersection(*base_vars_per_country.values())
    print(f"\nFound {len(common_base_vars)} common base variables across all datasets")

    processed_dict = {}
    for country, df in data_dict.items():
        keep_cols = [col for col in df.columns
                     if strip_country_code(col, country) in common_base_vars]
        rename_dict = {col: strip_country_code(col, country) for col in keep_cols}
        processed_dict[country] = df[keep_cols].rename(columns=rename_dict)

    return processed_dict


def save_as_3d_array(data_dict, date_col):
    """Saves as 3D NumPy array with shape (time × countries × variables)"""
    n_countries = len(data_dict)
    n_time = min(df.shape[0] for df in data_dict.values())
    n_variables = len(next(iter(data_dict.values())).columns) - 1  # Exclude date_col

    # Create array with shape (time, countries, variables)
    data_3d = np.zeros((n_time, n_countries, n_variables))

    country_list = sorted(data_dict.keys())
    variables = None

    # Fill the array
    for j, country in enumerate(country_list):
        df = data_dict[country]
        numeric_df = df.drop(columns=[date_col])
        # Assign data with time as first dimension
        data_3d[:, j, :] = numeric_df.values
        if j == 0:  # Store variables from first country without postfixes
            variables = numeric_df.columns.tolist()

    output_file = OUTPUT_DIR / 'processed_3d_data.npz'
    dates = data_dict[country_list[0]][date_col].values
    np.savez(
        output_file,
        data=data_3d,
        dates=dates,
        countries=country_list,
        variables=variables
    )

    print(f"\nSaved 3D array to {output_file}")
    print(f"Shape: {data_3d.shape} (time × countries × variables)")
    print(f"Countries: {country_list}")
    print(f"Variables: {variables}")


def main():
    print("Starting preprocessing...")

    data_dict = get_all_data()
    date_col, min_length = check_datasets(data_dict)
    data_dict = reshape_datasets(data_dict, date_col, min_length)
    data_dict = find_common_variables(data_dict)
    save_as_3d_array(data_dict, date_col)

    print("\nPreprocessing complete!")


def count_variables(data_dict, date_col):
    """Counts the number of variables (columns excluding date_col) in each country's dataset"""
    print("\nVariable Count per Country:")
    variable_counts = {}
    for country, df in data_dict.items():
        # Exclude the date column from the variable count
        num_variables = df.shape[1] - (1 if date_col in df.columns else 0)
        variable_counts[country] = num_variables
        print(f"{country}: {num_variables} variables")

    # Summary statistics
    print(f"\nSummary of Variable Counts:")
    print(f"Total datasets: {len(variable_counts)}")
    print(f"Minimum variables: {min(variable_counts.values())}")
    print(f"Maximum variables: {max(variable_counts.values())}")
    print(f"Average variables: {sum(variable_counts.values()) / len(variable_counts):.2f}")

if __name__ == "__main__":
    main()
