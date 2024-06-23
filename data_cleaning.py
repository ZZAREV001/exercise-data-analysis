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


def clean_data(df):
    # Assuming 'date' is the column to clean
    date_column = 'date'

    # Check if the 'date' column exists before attempting to clean it
    if date_column in df.columns:
        df = clean_date_column(df, date_column)
    else:
        print(f"Skipping cleaning 'date' column as it does not exist in the DataFrame.")

    # Add other cleaning steps here

    return df
