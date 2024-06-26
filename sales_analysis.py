import io
import pandas as pd
import matplotlib.pyplot as plt
import requests

from data_cleaning import clean_sales_method


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
    if 'revenue' in df.columns:
        total_sales = df['revenue'].sum()
        print(f"\nTotal sales: ${total_sales:,.2f}")
    else:
        print("Column 'revenue' not found in the data.")


def analyze_sales_by_category(df):
    """Calculates and prints sales grouped by sales method, with data cleaning."""
    if 'sales_method' in df.columns and 'revenue' in df.columns:
        # Clean and standardize the sales_method column
        df['sales_method'] = df['sales_method'].apply(clean_sales_method)

        # Group by the cleaned sales method and sum revenue
        sales_by_category = df.groupby('sales_method')['revenue'].sum().sort_values(ascending=False)

        print("\nSales by method:")
        print(sales_by_category.to_markdown(numalign='left', stralign='left'))

        # Calculate percentage of total revenue
        total_revenue = sales_by_category.sum()
        sales_by_category_pct = (sales_by_category / total_revenue * 100).round(2)

        print("\nPercentage of total revenue:")
        print(sales_by_category_pct.to_markdown(numalign='left', stralign='left'))

        # Visualize the sales by method
        plt.figure(figsize=(10, 6))
        sales_by_category.plot(kind='bar')
        plt.title('Total Revenue by Sales Method')
        plt.xlabel('Sales Method')
        plt.ylabel('Total Revenue')
        plt.xticks(rotation=45)
        for i, v in enumerate(sales_by_category):
            plt.text(i, v, f'${v:,.0f}\n({sales_by_category_pct[i]:.1f}%)', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

        # Print number of customers for each method
        customers_by_method = df.groupby('sales_method')['customer_id'].nunique().sort_values(ascending=False)
        print("\nNumber of customers by method:")
        print(customers_by_method.to_markdown(numalign='left', stralign='left'))
    else:
        print("Either column 'sales_method' or 'revenue' not found in the data.")


def analyze_revenue_over_time(df):
    """Analyzes and plots revenue trends over time for each sales method."""

    if 'week' in df.columns and 'revenue' in df.columns and 'sales_method' in df.columns:
        # Aggregate revenue by week and sales method
        weekly_revenue = df.groupby(['week', 'sales_method'])['revenue'].sum().unstack()

        # Calculate cumulative revenue for each method
        cumulative_revenue = weekly_revenue.cumsum()

        # Create subplots for weekly and cumulative revenue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot weekly revenue
        weekly_revenue.plot(ax=ax1, marker='o')
        ax1.set_title('Weekly Revenue by Sales Method')
        ax1.set_xlabel('Week Since Product Launch')
        ax1.set_ylabel('Revenue')
        ax1.grid(axis='y')
        ax1.legend(title='Sales Method')

        # Plot cumulative revenue
        cumulative_revenue.plot(ax=ax2, marker='x')
        ax2.set_title('Cumulative Revenue by Sales Method')
        ax2.set_xlabel('Week Since Product Launch')
        ax2.set_ylabel('Cumulative Revenue')
        ax2.grid(axis='y')
        ax2.legend(title='Sales Method')

        plt.tight_layout()
        plt.show()

        # Calculate and print total revenue for each method
        total_revenue = weekly_revenue.sum().sort_values(ascending=False)
        print("\nTotal Revenue by Sales Method:")
        print(total_revenue.to_markdown(numalign='left', stralign='left'))

        # Calculate and print average weekly revenue for each method
        avg_weekly_revenue = weekly_revenue.mean().sort_values(ascending=False)
        print("\nAverage Weekly Revenue by Sales Method:")
        print(avg_weekly_revenue.to_markdown(numalign='left', stralign='left'))

    else:
        print("One or more required columns ('week', 'revenue', 'sales_method') not found in the data.")


import pandas as pd
import numpy as np


def calculate_descriptive_statistics(df):
    """
    Calculates and prints detailed descriptive statistics for numerical columns.
    Also provides a summary of non-numeric columns.
    """
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    if not numeric_columns.empty:
        # Calculate statistics for numeric columns
        stats = df[numeric_columns].agg([
            'count', 'mean', 'std', 'min',
            lambda x: x.quantile(0.25),
            'median',
            lambda x: x.quantile(0.75),
            'max', 'skew', 'kurtosis'
        ]).T

        # Rename some columns for clarity
        stats = stats.rename(columns={
            '<lambda_0>': '25%',
            '<lambda_1>': '75%'
        })

        # Round all values to 2 decimal places
        stats = stats.round(2)

        print("\nDescriptive Statistics for Numeric Columns:")
        print(stats.to_markdown(numalign='left', stralign='left'))

        # Calculate and print correlation matrix
        corr_matrix = df[numeric_columns].corr().round(2)
        print("\nCorrelation Matrix:")
        print(corr_matrix.to_markdown(numalign='left', stralign='left'))

    else:
        print("No numeric columns found in the data.")

    if not non_numeric_columns.empty:
        # Provide summary for non-numeric columns
        print("\nSummary of Non-Numeric Columns:")
        for col in non_numeric_columns:
            unique_values = df[col].nunique()
            most_common = df[col].value_counts().nlargest(3)
            print(f"\n{col}:")
            print(f"  Unique values: {unique_values}")
            print("  Most common values:")
            print(most_common.to_markdown(numalign='left', stralign='left'))
    else:
        print("No non-numeric columns found in the data.")

    # Print overall dataset information
    print("\nDataset Information:")
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of columns: {len(df.columns)}")
    print("\nColumn Types:")
    print(df.dtypes.to_markdown(numalign='left', stralign='left'))

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing Values:")
        print(missing_values[missing_values > 0].to_markdown(numalign='left', stralign='left'))
    else:
        print("\nNo missing values found in the dataset.")


def identify_top_products(df, n=10):
    """Identifies and prints the top N products by sales amount."""
    product_column = 'sales_method'
    amount_column = 'revenue'

    if product_column in df.columns and amount_column in df.columns:
        top_products = df.groupby(product_column)[amount_column].sum().nlargest(n)
        print(f"\nTop {n} Products by Sales Amount:")
        print(top_products.to_markdown(numalign='left', stralign='left'))
    else:
        print(f"Either '{product_column}' or '{amount_column}' column not found in the data.")


def visualize_category_distribution(df):
    """Creates a pie chart to visualize the distribution of sales across categories."""
    category_column = 'sales_method'
    amount_column = 'revenue'

    if category_column in df.columns and amount_column in df.columns:
        category_sales = df.groupby(category_column)[amount_column].sum()
        plt.figure(figsize=(10, 8))
        plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', startangle=90)
        plt.title('Sales Distribution by Category')
        plt.axis('equal')
        plt.show()
    else:
        print(f"Either '{category_column}' or '{amount_column}' column not found in the data.")


def generate_report(df):
    """Generates a comprehensive report by calling all analysis functions."""
    explore_data(df)
    calculate_total_sales(df)
    analyze_sales_by_category(df)
    analyze_revenue_over_time(df)
    calculate_descriptive_statistics(df)
    identify_top_products(df)
    visualize_category_distribution(df)
