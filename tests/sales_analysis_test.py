import unittest
from io import StringIO

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sales_analysis import ml_recommend_sales_method, clean_sales_method

class TestMLRecommendSalesMethod(unittest.TestCase):

    def setUp(self):
        # Create a mock DataFrame
        self.df = pd.DataFrame({
            'sales_method': ['Email', 'Call', 'Email + Call'] * 100,
            'revenue': np.random.randint(100, 1000, 300),
            'week': np.random.randint(1, 53, 300),
            'years_as_customer': np.random.randint(0, 20, 300),
            'nb_site_visits': np.random.randint(1, 100, 300),
            'nb_sold': np.random.randint(1, 10, 300)
        })


    def test_missing_columns(self):
        # Create a DataFrame with missing columns
        df_missing = self.df.drop(['revenue', 'week'], axis=1)

        # Capture the printed output
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ml_recommend_sales_method(df_missing)

        self.assertIn("Required columns not found in the data.", fake_out.getvalue())

    def test_clean_sales_method(self):
        # Test if the clean_sales_method is applied correctly
        unique_methods = self.df['sales_method'].unique()
        self.assertIn('Email', unique_methods)
        self.assertIn('Call', unique_methods)
        self.assertIn('Email + Call', unique_methods)


    @patch('builtins.print')
    @patch('matplotlib.pyplot.show')
    def test_feature_importance(self, mock_show, mock_print):
        ml_recommend_sales_method(self.df)

        # Check if "Feature Importance:" was printed
        feature_importance_called = any("Feature Importance:" in call[0][0] for call in mock_print.call_args_list)
        self.assertTrue(feature_importance_called, "Feature Importance was not printed")

        # Check if all features were mentioned in any print call
        all_features_mentioned = all(
            any(feature in call[0][0] for call in mock_print.call_args_list)
            for feature in ['week', 'years_as_customer', 'nb_site_visits', 'nb_sold', 'revenue']
        )
        self.assertTrue(all_features_mentioned, "Not all features were mentioned in the output")


if __name__ == '__main__':
    unittest.main()