import io
import requests
import pandas as pd


def read_sales_data_from_s3(bucket_name, object_key):
    """Reads sales data from a public S3 bucket into a Pandas DataFrame."""
    url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
    try:
        response = requests.get(url, verify=False)  # Temporarily disable SSL verification
        response.raise_for_status()

        # Wrap response content in a file-like object
        with io.StringIO(response.content.decode('utf-8')) as f:
            df = pd.read_csv(f)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from S3: {e}")
        return None


def explore_data(df):
    """Displays basic information and summary statistics about the DataFrame."""
    print("First 5 rows:")
    print(df.head().to_markdown(index=False, numalign='left', stralign='left'))
    print(df.info())


def calculate_total_sales(df):
    """Calculates and prints the total sales amount."""
    column_name = 'sales_amount'  # Use a variable for the column name
    matching_columns = [col for col in df.columns if col.lower() == column_name.lower()]

    if matching_columns:
        # Use the first matching column if there are multiple
        df = df.rename(columns={matching_columns[0]: column_name})
        total_sales = df[column_name].sum()
        print(f"\nTotal sales: ${total_sales:,.2f}")
    else:
        print(f"Column '{column_name}' not found in the data.")


def analyze_sales_by_category(df):
    """Calculates and prints sales grouped by category."""
    amount_column_name = 'sales_amount'  # Use a variable for the sales amount column name
    category_column_name = 'category'  # Use a variable for the category column name

    # Case-insensitive search for both columns
    matching_amount_columns = [col for col in df.columns if col.lower() == amount_column_name.lower()]
    matching_category_columns = [col for col in df.columns if col.lower() == category_column_name.lower()]

    if matching_amount_columns and matching_category_columns:
        # Use the first matching column for each if there are multiple
        df = df.rename(columns={
            matching_amount_columns[0]: amount_column_name,
            matching_category_columns[0]: category_column_name
        })

        sales_by_category = df.groupby(category_column_name)[amount_column_name].sum()
        print("\nSales by category:")
        print(sales_by_category.to_markdown(numalign='left', stralign='left'))
    else:
        print(f"Either column '{amount_column_name}' or '{category_column_name}' not found in the data.")

