import sys
import argparse
from data_cleaning import clean_data, validate_data_types
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
    calculate_revenue_per_customer,
    generate_report,
)


def main():
    parser = argparse.ArgumentParser(description="Sales Data Analysis")
    parser.add_argument('--analysis_type', type=int, required=True,
                        help="Number of the analysis to perform (1-11)")
    parser.add_argument('--top_n', type=int, default=10,
                        help="Number of top products to identify (used with analysis type 6)")

    args = parser.parse_args()

    # File path (S3 bucket and object key)
    bucket_name = "talent-assets.datacamp.com"
    object_key = "product_sales.csv"

    # Read the data using the updated function
    result = read_sales_data_from_s3(bucket_name, object_key)

    # Check if the result is a tuple (DataFrame, validation_results)
    if isinstance(result, tuple) and len(result) == 2:
        df, validation_results = result
    else:
        print("Unexpected return format from read_sales_data_from_s3. Please check the function.")
        sys.exit(1)

    # Check if the DataFrame was read successfully
    if df is not None:
        print("Data read successfully. Here's a preview of the data:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())

        # Check validation results
        if all(validation_results.values()):
            print("\nAll data validation checks passed.")
        else:
            print("\nWarning: Some data validation checks failed. Proceed with caution.")
            print("Failed validations:")
            for column, is_valid in validation_results.items():
                if not is_valid:
                    print(f"- {column}")

            proceed = input("Do you want to proceed with the analysis? (y/n): ").lower().strip()
            if proceed != 'y':
                print("Exiting the program. Please check the data and try again.")
                sys.exit(1)

        # Clean the data
        print("\nCleaning the data...")
        df = clean_data(df)
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
                calculate_revenue_per_customer(df)
            elif args.analysis_type == 12:
                generate_report(df)
            else:
                print("Invalid analysis type. Please enter a number between 1 and 11.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")
    else:
        print("Failed to read data. Please check the S3 bucket and object key and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()