The command-line terminal is used to run the data
analysis progressively:
python main.py --analysis_type 'number of the analysis we want'

1. Explore the data:
python main.py --analysis_type 1

2. Calculate Total Sales:
python main.py --analysis_type 2

3. Analyze Sales by Category:
python main.py --analysis_type 3

4. Analyze Sales Over Time:
python main.py --analysis_type 4

5. Calculate Descriptive Statistics:
The --top_n argument is optional and only needed if you are using analysis type 6. If not provided, it defaults to 10. For example, to identify the top 5 products, you would run:
python main.py --analysis_type 5

6. Identify Top Products:
python main.py --analysis_type 6 --top_n 5

7. Visualize Category Distribution:
python main.py --analysis_type 7

8. Visualize the sales volume profile:
python main.py --analysis_type 8

9. Analyze the revenue distribution:
python main.py --analysis_type 9

10. Predict the most effective sales method:
python main.py --analysis_type 10

11. Business metric: Calculates the Revenue per Customer metric for each sales method over time.
python main.py --analysis_type 11

12. Generate a report:
python main.py --analysis_type 12

13. Exit the program:
python main.py --analysis_type 13
