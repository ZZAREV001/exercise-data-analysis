import pandas as pd
import numpy as np
from datetime import datetime


def clean_date_column(df, column_name='date'):
    """
    Cleans the date column by converting it to datetime format and handling any invalid dates.
    """
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    # Drop rows with invalid dates
    df.dropna(subset=[column_name], inplace=True)
    return df


def clean_numeric_column(df, column_name):
    """
    Cleans a numeric column by removing non-numeric characters and converting to float.
    """
    df[column_name] = df[column_name].replace('[\$,]', '', regex=True).astype(float)
    return df


def clean_category_column(df, column_name='category'):
    """
    Cleans the category column by standardizing category names.
    """
    df[column_name] = df[column_name].str.strip().str.lower()
    return df


def clean_product_column(df, column_name='product'):
    """
    Cleans the product column by removing any special characters and standardizing format.
    """
    df[column_name] = df[column_name].str.strip().str.replace('[^\w\s]', '', regex=True)
    return df


def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()


def handle_missing_values(df):
    """
    Handles missing values in the DataFrame.
    """
    # For numeric columns, fill missing values with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # For categorical columns, fill missing values with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

    return df


def validate_data_types(df: pd.DataFrame) -> dict:
    """
    Validates the data types of each column in the DataFrame.
    Returns a dictionary of column names and their validation status.
    """
    expected_types = {
        'week': [np.int64, np.int32],
        'sales_method': [object, pd.StringDtype()],
        'customer_id': [object, pd.StringDtype()],
        'nb_sold': [np.int64, np.int32],
        'revenue': [np.float64, np.float32],
        'years_as_customer': [np.int64, np.int32],
        'nb_site_visits': [np.int64, np.int32],
        'state': [object, pd.StringDtype()]
    }

    validation_results = {}

    for column, expected_type_list in expected_types.items():
        if column in df.columns:
            is_valid = type(df[column].dtype) in expected_type_list or df[column].dtype in expected_type_list
            validation_results[column] = is_valid
            if not is_valid:
                print(f"Warning: Column '{column}' has type {df[column].dtype}, expected one of {expected_type_list}")
        else:
            validation_results[column] = False
            print(f"Warning: Column '{column}' is missing from the DataFrame")

    return validation_results


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data in the DataFrame.
    """
    # Week column
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    df = df[df['week'].notnull() & (df['week'] >= 0)]
    print("Cleaned 'week' column.")

    # Sales method column
    df['sales_method'] = df['sales_method'].apply(clean_sales_method)
    df = df[df['sales_method'] != 'unknown']
    print("Cleaned 'sales_method' column.")

    # Customer ID column
    df['customer_id'] = df['customer_id'].astype(str).str.strip()
    df = df[df['customer_id'] != '']
    print("Cleaned 'customer_id' column.")

    # Number of products sold column
    df['nb_sold'] = pd.to_numeric(df['nb_sold'], errors='coerce')
    df = df[df['nb_sold'].notnull() & (df['nb_sold'] >= 0)]
    print("Cleaned 'nb_sold' column.")

    # Revenue column
    df['revenue'] = pd.to_numeric(df['revenue'].replace('[\$,]', '', regex=True), errors='coerce')
    df = df[df['revenue'].notnull() & (df['revenue'] >= 0)]
    df['revenue'] = df['revenue'].round(2)
    print("Cleaned 'revenue' column.")

    # Years as customer column
    df['years_as_customer'] = pd.to_numeric(df['years_as_customer'], errors='coerce')
    max_years = datetime.now().year - 1984  # Company founded in 1984
    df = df[df['years_as_customer'].notnull() & (df['years_as_customer'] >= 0) & (df['years_as_customer'] <= max_years)]
    print("Cleaned 'years_as_customer' column.")

    # Number of site visits column
    df['nb_site_visits'] = pd.to_numeric(df['nb_site_visits'], errors='coerce')
    df = df[df['nb_site_visits'].notnull() & (df['nb_site_visits'] >= 0)]
    print("Cleaned 'nb_site_visits' column.")

    # State column
    df['state'] = df['state'].str.strip().str.upper()
    print("Cleaned 'state' column.")

    # Remove duplicates
    original_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = original_rows - len(df)
    print(f"Removed {removed_rows} duplicate rows.")

    return df


def clean_sales_method(method):
    """Standardize sales method names"""
    method = str(method).lower().strip()
    if method in ['em + call', 'email + call']:
        return 'email + call'
    elif method in ['email', 'em']:
        return 'email'
    elif method == 'call':
        return 'call'
    else:
        return 'unknown'


# Printing the summary of the cleaning process
def print_cleaning_summary(df_original, df_cleaned):
    """Print a summary of the changes made during the cleaning process"""
    print("\nData Cleaning Summary:")
    print(f"Original rows: {len(df_original)}")
    print(f"Cleaned rows: {len(df_cleaned)}")
    print(f"Rows removed: {len(df_original) - len(df_cleaned)}")
    print("\nUnique values in 'sales_method' after cleaning:")
    print(df_cleaned['sales_method'].value_counts().to_markdown())
    print("\nMissing values after cleaning:")
    print(df_cleaned.isnull().sum().to_markdown())
    print("\nData types after cleaning:")
    print(df_cleaned.dtypes.to_markdown())
