import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import requests
from typing import Optional, Tuple

from data_cleaning import clean_sales_method
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def validate_data_types(df: pd.DataFrame) -> dict:
    """
    Validates the data types of each column in the DataFrame.
    Returns a dictionary of column names and their validation status.
    """
    expected_types = {
        'week': 'int64',
        'sales_method': 'object',
        'customer_id': 'object',
        'nb_sold': 'int64',
        'revenue': 'float64',
        'years_as_customer': 'int64',
        'nb_site_visits': 'int64',
        'state': 'object'
    }

    validation_results = {}

    for column, expected_type in expected_types.items():
        if column in df.columns:
            is_valid = df[column].dtype == expected_type
            validation_results[column] = is_valid
            if not is_valid:
                print(f"Warning: Column '{column}' has type {df[column].dtype}, expected {expected_type}")
        else:
            validation_results[column] = False
            print(f"Warning: Column '{column}' is missing from the DataFrame")

    return validation_results


def read_sales_data_from_s3(bucket_name: str, object_key: str) -> Tuple[Optional[pd.DataFrame], dict]:
    """
    Reads sales data from a public S3 bucket into a Pandas DataFrame and validates its structure.
    Returns a tuple containing the DataFrame (or None if an error occurred) and the validation results.
    """
    url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
    try:
        response = requests.get(url, verify=False)  # Temporarily disable SSL verification
        response.raise_for_status()

        # Wrap response content in a file-like object
        with io.StringIO(response.content.decode('utf-8')) as f:
            df = pd.read_csv(f)

        # Validate the data types
        validation_results = validate_data_types(df)

        if all(validation_results.values()):
            print("All columns present and have correct types.")
        else:
            print("Some columns are missing or have incorrect types. Data may need cleaning or further investigation.")

        return df, validation_results
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from S3: {e}")
        return None, {}


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
        plt.figure(figsize=(12, 8))
        bars = sales_by_category.plot(kind='bar')
        plt.title('Total Revenue by Sales Method')
        plt.xlabel('Sales Method')
        plt.ylabel('Total Revenue')
        plt.xticks(rotation=45)

        # Set y-limits to accommodate text labels
        plt.ylim(0, max(sales_by_category) * 1.2)

        for i, v in enumerate(sales_by_category):
            plt.text(i, v + (0.05 * total_revenue), f'${v:,.0f}\n({sales_by_category_pct[i]:.1f}%)', ha='center',
                     va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

        # Print number of customers for each method
        customers_by_method = df.groupby('sales_method')['customer_id'].nunique().sort_values(ascending=False)
        print("\nNumber of customers by method:")
        print(customers_by_method.to_markdown(numalign='left', stralign='left'))

        # Add a visualization for number of customers by method
        plt.figure(figsize=(12, 8))
        customers_by_method.plot(kind='bar')
        plt.title('Number of Customers by Sales Method')
        plt.xlabel('Sales Method')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45)

        # Set y-limits to accommodate text labels
        plt.ylim(0, max(customers_by_method) * 1.2)

        for i, v in enumerate(customers_by_method):
            plt.text(i, v + (0.05 * max(customers_by_method)), f'{v}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

    else:
        print("Either column 'sales_method' or 'revenue' not found in the data.")


def analyze_revenue_over_time(df):
    """Analyzes and plots revenue trends over time for each sales method."""

    if 'week' in df.columns and 'revenue' in df.columns and 'sales_method' in df.columns:
        # Ensure sales_method is cleaned
        df['sales_method'] = df['sales_method'].apply(clean_sales_method)

        # Aggregate revenue by week and sales method
        weekly_revenue = df.groupby(['week', 'sales_method'])['revenue'].sum().unstack()

        # Calculate cumulative revenue for each method
        cumulative_revenue = weekly_revenue.cumsum()

        # Create subplots for weekly and cumulative revenue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})

        # Plot weekly revenue
        weekly_revenue.plot(ax=ax1, marker='o')
        ax1.set_title('Weekly Revenue by Sales Method', fontsize=16, pad=20)
        ax1.set_xlabel('Week Since Product Launch', fontsize=12)
        ax1.set_ylabel('Revenue', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend(title='Sales Method', fontsize=10, title_fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # Plot cumulative revenue
        cumulative_revenue.plot(ax=ax2, marker='x')
        ax2.set_title('Cumulative Revenue by Sales Method', fontsize=16, pad=20)
        ax2.set_xlabel('Week Since Product Launch', fontsize=12)
        ax2.set_ylabel('Cumulative Revenue', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.legend(title='Sales Method', fontsize=10, title_fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=10)

        # Adjust layout and add padding
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95, hspace=0.4)

        plt.show()

        # [Rest of the function remains the same]

    else:
        print("One or more required columns ('week', 'revenue', 'sales_method') not found in the data.")


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


def plot_sales_volume_profile(df):
    """
    Creates a plot similar to a volume profile, showing the distribution of sales across weeks.
    """
    if 'week' in df.columns and 'revenue' in df.columns:
        # Group data by week and calculate total revenue
        weekly_sales = df.groupby('week')['revenue'].sum().sort_index()

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot horizontal bars
        bars = ax.barh(weekly_sales.index, weekly_sales.values, height=0.8, alpha=0.7)

        # Customize the plot
        ax.set_title('Sales Volume Profile', fontsize=16)
        ax.set_xlabel('Total Revenue', fontsize=12)
        ax.set_ylabel('Week', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Add value labels to the end of each bar with an offset
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 5000  # Offset by 5000 for better spacing
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                    f'${width:,.0f}', va='center', fontsize=10, color='black')

        # Invert y-axis to have earlier weeks at the top
        ax.invert_yaxis()

        # Add a vertical line for the mean revenue
        mean_revenue = weekly_sales.mean()
        ax.axvline(mean_revenue, color='r', linestyle='--', label=f'Mean: ${mean_revenue:,.0f}')

        # Add legend
        ax.legend()

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\nSales Volume Profile Summary:")
        print(f"Total Weeks: {len(weekly_sales)}")
        print(f"Total Revenue: ${weekly_sales.sum():,.2f}")
        print(f"Average Weekly Revenue: ${weekly_sales.mean():,.2f}")
        print(f"Highest Revenue Week: Week {weekly_sales.idxmax()} (${weekly_sales.max():,.2f})")
        print(f"Lowest Revenue Week: Week {weekly_sales.idxmin()} (${weekly_sales.min():,.2f})")
    else:
        print("Either 'week' or 'revenue' column not found in the data.")


def analyze_revenue_distribution(df):
    """
    Analyzes and visualizes the spread of revenue overall and for each sales method.
    """
    if 'revenue' in df.columns and 'sales_method' in df.columns:
        # Ensure sales_method is cleaned
        df['sales_method'] = df['sales_method'].apply(clean_sales_method)

        # Create a figure with two subplots, increase figure size and add space between subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'hspace': 0.3})

        # Overall revenue distribution
        sns.boxplot(x=df['revenue'], ax=ax1)
        ax1.set_title('Overall Revenue Distribution', fontsize=16, pad=20)
        ax1.set_xlabel('Revenue', fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # Revenue distribution by sales method
        sns.boxplot(x='revenue', y='sales_method', data=df, ax=ax2)
        ax2.set_title('Revenue Distribution by Sales Method', fontsize=16, pad=20)
        ax2.set_xlabel('Revenue', fontsize=12)
        ax2.set_ylabel('Sales Method', fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=10)

        # Adjust layout and add padding
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95)

        plt.show()

    else:
        print("Either 'revenue' or 'sales_method' column not found in the data.")


def ml_recommend_sales_method(df):
    """
    Uses machine learning (Random Forest) to predict the most effective sales method
    based on customer characteristics and transaction details.
    """
    if all(col in df.columns for col in
           ['sales_method', 'revenue', 'week', 'years_as_customer', 'nb_site_visits', 'nb_sold']):
        # Prepare the data
        df['sales_method'] = df['sales_method'].apply(clean_sales_method)

        # Select features and target
        features = ['week', 'years_as_customer', 'nb_site_visits', 'nb_sold', 'revenue']
        X = df[features]
        y = df['sales_method']

        # Encode the target variable
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_classifier.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print("\nMachine Learning Model Performance:")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance.to_markdown(index=False, numalign='left', stralign='left'))

        plt.figure(figsize=(10, 6))
        sns.barplot(x='feature', y='importance', data=feature_importance)
        plt.title('Feature Importance in Predicting Sales Method')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Predict probabilities for each sales method
        method_probs = rf_classifier.predict_proba(X)
        df_probs = pd.DataFrame(method_probs, columns=le.classes_)

        # Add these probabilities to the original dataframe
        df = pd.concat([df, df_probs], axis=1)

        # Analyze which method is most effective for different customer segments
        print("\nAverage Probability of Success for Each Method by Customer Tenure:")
        tenure_groups = pd.cut(df['years_as_customer'], bins=[0, 1, 5, 10, np.inf],
                               labels=['0-1 years', '1-5 years', '5-10 years', '10+ years'])
        print(df.groupby(tenure_groups, observed=True)[le.classes_].mean().round(2).to_markdown(numalign='left',
                                                                                                stralign='left'))

        print("\nRecommendations based on Machine Learning Analysis:")
        print("1. Overall Best Method:", le.classes_[np.argmax(method_probs.mean(axis=0))])
        print("2. Best Method for New Customers (0-1 years):",
              le.classes_[np.argmax(df[df['years_as_customer'] <= 1][le.classes_].mean())])
        print("3. Best Method for Long-term Customers (10+ years):",
              le.classes_[np.argmax(df[df['years_as_customer'] > 10][le.classes_].mean())])

        print("\nConsiderations:")
        print("- The model's accuracy indicates how reliable these predictions are.")
        print("- Consider the feature importance when deciding which factors to prioritize in our sales strategy.")
        print(
            "- The effectiveness of each method varies for different customer segments, suggesting a tailored approach might be beneficial.")
        print(
            "- This analysis should be combined with the previous metrics and practical considerations like team time investment.")

    else:
        print("Required columns not found in the data.")


def calculate_revenue_per_customer(df):
    """
    Calculates the Revenue per Customer metric for each sales method over time.
    Args:
    df (pandas.DataFrame): The sales data DataFrame
    Returns:
    pandas.DataFrame: A DataFrame with Revenue per Customer for each sales method and week
    """
    # Ensure sales_method is cleaned
    df['sales_method'] = df['sales_method'].apply(clean_sales_method)

    # Group by week and sales method
    grouped = df.groupby(['week', 'sales_method'])

    # Calculate total revenue and number of unique customers for each group
    revenue = grouped['revenue'].sum()
    customers = grouped['customer_id'].nunique()

    # Calculate revenue per customer
    revenue_per_customer = revenue / customers

    # Reshape the data for easier plotting
    revenue_per_customer = revenue_per_customer.unstack()

    # Plot the results
    plt.figure(figsize=(12, 6))
    revenue_per_customer.plot(marker='o')
    plt.title('Revenue per Customer by Sales Method Over Time')
    plt.xlabel('Week')
    plt.ylabel('Revenue per Customer ($)')
    plt.legend(title='Sales Method')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nRevenue per Customer Summary:")
    print(revenue_per_customer.describe().to_markdown(numalign='left', stralign='left'))

    # Calculate overall average for each method
    overall_average = revenue_per_customer.mean()
    print("\nOverall Average Revenue per Customer by Sales Method:")
    print(overall_average.to_markdown(numalign='left', stralign='left'))

    return revenue_per_customer

def generate_report(df):
    """Generates a comprehensive report by calling all analysis functions."""
    explore_data(df)
    calculate_total_sales(df)
    print("\n--- Answering: How many customers were there for each approach? ---")
    analyze_sales_by_category(df)
    analyze_revenue_over_time(df)
    calculate_descriptive_statistics(df)
    identify_top_products(df)
    visualize_category_distribution(df)
    print("\n--- Answering: What does the spread of the revenue look like overall? And for each method? ---")
    analyze_revenue_distribution(df)
    ml_recommend_sales_method(df)
    plot_sales_volume_profile(df)
    print("\n--- Analyzing Revenue per Customer Metric ---")
    calculate_revenue_per_customer(df)
