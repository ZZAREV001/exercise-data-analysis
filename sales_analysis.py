import io
import pandas as pd
import matplotlib.pyplot as plt
import requests


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
    column_name = 'week'  # Use a variable for the column name
    if column_name in df.columns:
        total_sales = df[column_name].sum()
        print(f"\nTotal sales: ${total_sales:,.2f}")
    else:
        print(f"Column '{column_name}' not found in the data.")


def analyze_sales_by_category(df):
    """Calculates and prints sales grouped by category."""
    amount_column_name = 'sales_method'
    category_column_name = 'customer_id'

    if amount_column_name in df.columns and category_column_name in df.columns:
        sales_by_category = df.groupby(category_column_name)[amount_column_name].sum()
        print("\nSales by category:")
        print(sales_by_category.to_markdown(numalign='left', stralign='left'))
    else:
        print(f"Either column '{amount_column_name}' or '{category_column_name}' not found in the data.")


def analyze_revenue_over_time(df):
    """Analyzes and plots revenue trends over time using the 'week' column."""

    # Check if 'week' and 'revenue' columns exist
    if 'week' in df.columns and 'revenue' in df.columns:
        # Aggregate revenue by week and calculate cumulative revenue
        weekly_revenue = df.groupby('week')['revenue'].sum().reset_index()
        weekly_revenue['cumulative_revenue'] = weekly_revenue['revenue'].cumsum()

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_revenue['week'], weekly_revenue['revenue'], marker='o', linestyle='-')  # Line plot
        plt.plot(weekly_revenue['week'], weekly_revenue['cumulative_revenue'], marker='x',
                 linestyle='--')  # Added cumulative line

        # Add labels and title
        plt.title('Revenue Trend Over Time')
        plt.xlabel('Week Since Product Launch')
        plt.ylabel('Revenue')
        plt.grid(axis='y')  # Add a grid to the y-axis
        plt.legend(['Weekly Revenue', 'Cumulative Revenue'])  # Added legend for two lines

        # Show the plot
        plt.show()
    else:
        print(f"Either column 'week' or 'revenue' not found in the data.")


def calculate_descriptive_statistics(df):
    """Calculates and prints descriptive statistics for numerical columns."""
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_columns.empty:
        stats = df[numeric_columns].describe()
        print("\nDescriptive Statistics:")
        print(stats.to_markdown(numalign='left', stralign='left'))
    else:
        print("No numeric columns found in the data.")


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
