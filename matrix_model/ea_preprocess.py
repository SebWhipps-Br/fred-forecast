from pathlib import Path

import numpy as np
import pandas as pd

# Defines the directory path
BASE_DIR = Path('EA-MD-HT')
OUTPUT_DIR = Path('processed_data')
OUTPUT_DIR.mkdir(exist_ok=True)


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

    if date_col:
        common_dates = set.intersection(*[set(df[date_col]) for df in data_dict.values()])
        common_dates = sorted(list(common_dates))[:min_length]
        for country, df in data_dict.items():
            processed_dict[country] = df[df[date_col].isin(common_dates)].sort_values(date_col)
    else:
        for country, df in data_dict.items():
            processed_dict[country] = df.iloc[:min_length]

    return processed_dict


def find_common_variables(data_dict):
    """Removes variables not common across all datasets, importantly handling country postfixes"""

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
        # Renames columns to strip postfixes
        rename_dict = {col: strip_country_code(col, country) for col in keep_cols}
        processed_dict[country] = df[keep_cols].rename(columns=rename_dict)

    return processed_dict


def save_as_3d_array(data_dict, date_col):
    """Saves as 3D NumPy array"""
    n_countries = len(data_dict)
    n_time = min(df.shape[0] for df in data_dict.values())
    n_variables = len(next(iter(data_dict.values())).columns) - (1 if date_col else 0)

    data_3d = np.zeros((n_countries, n_time, n_variables))

    country_list = sorted(data_dict.keys())
    variables = None

    for i, country in enumerate(country_list):
        df = data_dict[country]
        if date_col:
            numeric_df = df.drop(columns=[date_col])
        else:
            numeric_df = df
        data_3d[i] = numeric_df.values
        if i == 0:  # Stores variables from first country without postfixes
            variables = numeric_df.columns.tolist()

    output_file = OUTPUT_DIR / 'processed_3d_data.npz'
    if date_col:
        dates = data_dict[country_list[0]][date_col].values
        np.savez(
            output_file,
            data=data_3d,
            dates=dates,
            countries=country_list,
            variables=variables
        )
    else:
        np.savez(
            output_file,
            data=data_3d,
            countries=country_list,
            variables=variables
        )

    print(f"\nSaved 3D array to {output_file}")
    print(f"Shape: {data_3d.shape} (countries × time × variables)")
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


if __name__ == "__main__":
    main()
