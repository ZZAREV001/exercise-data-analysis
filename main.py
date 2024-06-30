import sys
import argparse
from data_cleaning import (
    clean_data,
    remove_duplicates,
    handle_missing_values
)

from sales_analysis import (
    read_sales_data_from_s3,
    explore_data,
    calculate_total_sales,
    analyze_sales_by_category,
    analyze_revenue_over_time,
    calculate_descriptive_statistics,
    identify_top_products,
    visualize_category_distribution,
    plot_sales_volume_profile,
    analyze_revenue_distribution,
    ml_recommend_sales_method,
    generate_report,
)


def main():
    parser = argparse.ArgumentParser(description="Sales Data Analysis")
    parser.add_argument('--analysis_type', type=int, required=True,
                        help="Number of the analysis to perform (1-9)")
    parser.add_argument('--top_n', type=int, default=10,
                        help="Number of top products to identify (used with analysis type 6)")

    args = parser.parse_args()

    # File path (URL)
    file_path = "talent-assets.datacamp.com/product_sales.csv"

    # Get the bucket name and object key
    bucket_name = file_path.split("/")[0]
    object_key = "/".join(file_path.split("/")[1:])

    # Read the data using the updated function
    df = read_sales_data_from_s3(bucket_name, object_key)

    # Check if the DataFrame was read successfully
    if df is not None:
        print("Data read successfully. Here's a preview of the data:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())

        # Clean the data
        print("\nCleaning the data...")
        df = clean_data(df)
        df = remove_duplicates(df)
        df = handle_missing_values(df)
        print("Data cleaning completed.")

        print(f"You selected option: {args.analysis_type}")  # Debug print

        try:
            if args.analysis_type == 1:
                explore_data(df)
            elif args.analysis_type == 2:
                calculate_total_sales(df)
            elif args.analysis_type == 3:
                analyze_sales_by_category(df)
            elif args.analysis_type == 4:
                analyze_revenue_over_time(df)
            elif args.analysis_type == 5:
                calculate_descriptive_statistics(df)
            elif args.analysis_type == 6:
                identify_top_products(df, args.top_n)
            elif args.analysis_type == 7:
                visualize_category_distribution(df)
            elif args.analysis_type == 8:
                plot_sales_volume_profile(df)
            elif args.analysis_type == 9:
                analyze_revenue_distribution(df)
            elif args.analysis_type == 10:
                ml_recommend_sales_method(df)
            elif args.analysis_type == 11:
                generate_report(df)
            elif args.analysis_type == 12:
                print("Exiting the program. Goodbye!")
            else:
                print("Invalid analysis type. Please enter a number between 1 and 9.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")
    else:
        print("Failed to read data. Please check the file path and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
