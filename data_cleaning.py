import pandas as pd
import numpy as np


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


def clean_sales_method(method):
    """Standardize sales method names"""
    method = method.lower().strip()
    if method in ['em + call', 'email + call']:
        return 'email + call'
    elif method in ['email', 'em']:
        return 'email'
    elif method == 'call':
        return 'call'
    else:
        return 'unknown'  # Handle any unexpected values


def clean_data(df):
    # Clean date column
    if 'date' in df.columns:
        df = clean_date_column(df, 'date')
    else:
        print("Skipping cleaning 'date' column as it does not exist in the DataFrame.")

    # Clean numeric columns
    numeric_columns = ['revenue', 'nb_sold', 'years_as_customer', 'nb_site_visits']
    for col in numeric_columns:
        if col in df.columns:
            df = clean_numeric_column(df, col)
        else:
            print(f"Skipping cleaning '{col}' column as it does not exist in the DataFrame.")

    # Clean sales method column
    if 'sales_method' in df.columns:
        df['sales_method'] = df['sales_method'].apply(clean_sales_method)
        print("Cleaned 'sales_method' column.")
    else:
        print("Skipping cleaning 'sales_method' column as it does not exist in the DataFrame.")

    # Clean state column
    if 'state' in df.columns:
        df['state'] = df['state'].str.strip().str.upper()
        print("Cleaned 'state' column.")
    else:
        print("Skipping cleaning 'state' column as it does not exist in the DataFrame.")

    # Remove duplicates
    original_rows = len(df)
    df = remove_duplicates(df)
    removed_rows = original_rows - len(df)
    print(f"Removed {removed_rows} duplicate rows.")

    # Handle missing values
    df = handle_missing_values(df)
    print("Handled missing values.")

    # Remove rows with 'unknown' sales method
    unknown_rows = df[df['sales_method'] == 'unknown']
    if not unknown_rows.empty:
        print(f"Removing {len(unknown_rows)} rows with unknown sales method.")
        df = df[df['sales_method'] != 'unknown']

    return df


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
