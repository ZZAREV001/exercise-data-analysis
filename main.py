from sales_analysis import read_sales_data_from_s3, explore_data, calculate_total_sales, analyze_sales_by_category

# File path (URL)
file_path = "talent-assets.datacamp.com/product_sales.csv"

# Get the bucket name and object key (same as before)
bucket_name = file_path.split("/")[0]
object_key = "/".join(file_path.split("/")[1:])

# Read the data using the updated function
df = read_sales_data_from_s3(bucket_name, object_key)

# Check if the DataFrame was read successfully
if df is not None:
    # Proceed with data analysis
    while True:
        analysis_type = input("Enter the type of analysis you want (explore, total, category, or exit): ")

        if analysis_type.lower() == 'explore':
            explore_data(df)
        elif analysis_type.lower() == 'total':
            calculate_total_sales(df)
        elif analysis_type.lower() == 'category':
            analyze_sales_by_category(df)
        elif analysis_type.lower() == 'exit':
            break
        else:
            print("Invalid analysis type.")
else:
    print("Could not read the sales data. Please check the file format and ensure it has the correct structure.")

    # Exit if df is None
    exit()  # or sys.exit() for a more explicit exit
