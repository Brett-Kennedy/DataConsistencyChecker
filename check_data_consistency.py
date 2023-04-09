import pandas as pd
import numpy as np
import numbers
import os
import sys
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score
import copy
import math
import statistics
import datetime
import time
from dateutil.relativedelta import relativedelta
import pandas.api.types as pandas_types
import random
import string
import scipy
from itertools import combinations
from IPython import get_ipython
from IPython.display import display, Markdown
from textwrap import wrap
from termcolor import colored
import concurrent
from multiprocessing import Process, Queue

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Display options. There are relevant only when running this in a debugger or notebook.
pd.set_option('display.width', 32000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_colwidth', 3000)
pd.set_option('display.max_rows', 5000)

# Constants used to generate synthetic data
letters = string.ascii_letters
digits = string.digits
alphanumeric = letters + digits

# Constants used to define the tests. The tests are defined in a dictionary. These specify the index of the 
TEST_DEFN_DESC = 0          # Element 0 provides a description of the test
TEST_DEFN_FUNC = 1          # Element 1 specifies the function used to execute the test
TEST_DEFN_GEN_FUNC = 2      # Element 2 specifies the function used to generate demo data for the test
TEST_DEFN_SHORTLIST = 3     # Element 3 indicates if the test is in the shortlist of tests, which are the tests returned
                            #   by default when listing patterns found without exceptions.
TEST_DEFN_IMPLEMENTED = 4   # Element 4 specifies if the test is implemented
TEST_DEFN_FAST = 5          # Element 5 indicates if the test is reasonably fast even with thousands of columns.

# todo:
#   test all methods with Nones (Test_DC_Synthetic has a cell to test this)
#   we may want to add the comcept of minimum_valid_rows -- any test needs to have this many rows have non-missing values
#       in all columns. For example, if there are 1000 rows and min_valid_rows is 800, then when running LARGER, we
#       can only run this test on pairs of columns whwere at least 800 rows have non-missing values in both columns
#   many tests should check correlation before declaring a patern. eg sum of columns, mean of columsn, larger than sum
#   many tests flag exceptions just over some limit. eg much larger, which checks most values in a are 10x the values
#       in b, but it may flag things that are 9.99x.
#   test the large / small values tests with lognormal, exponantial, and other distributions. Can do in a notebook - just
#       generate diff synth data and see what it finds. Or mangle real data from OpenML with very small / large / negative
#       inverted, etc values.


class DataConsistencyChecker:
    def __init__(self,
                 execute_list=None,
                 exclude_list=None,
                 iqr_limit=3.5,
                 idr_limit=1.0,
                 max_combinations=1_000_000,
                 verbose=1):
        """
        execute_list: a list of test ids. If specified, only these tests will be executed.
        exclude_list: a list of test ids. If specified, these tests will not be executed; all others will be.
        iqr_limit: Inter-quartile range is used in several tests to find unusually small or large values. For example, to
                    identify large values, Q3 (the 3rd quartile, or 75th percentile plus some multiplier times the
                    inter-quartile range is often used. Ex: Q3 + 1.5*IQR. To avoid noisy results, a higher coefficient
                    is used here.
        idr_limit: Inter-decile range is used in several tests, particularly to identify very small values, as IQR
                    can work poorly where the values are strictly positive.
        max_combinations: Several tests check many combinations of columns, which can be very slow to execute where
                    there are many columns. For example, BIN_NUM_SAME will check subsets of the binary columns,
                    testing different sizes of subsets. For some sizes of subsets, there may be a very large number of
                    combinations. Setting max_combinations will restrict the number of combinations examined. This may
                    result in missing some patterns, but allows execution time to be limited. This applies only to
                    tests that check subsets of multiple sizes.
        verbose:    If 0, no output will be displayed until the tests are complete.
                    If 1, the test names will be displayed as they execute.
                    If 2, progress related to each of the more expensive tests will be displayed as well.
        """

        # execute_list and exclude_list should not both be set.
        assert execute_list is None or exclude_list is None

        # The set of tests that are to be run / not run may optionally be specified. This may also effect any synthetic
        # data which is generated.
        self.execute_list = execute_list
        self.exclude_list = exclude_list
        self.execution_test_list = []  # The complete list for one call to check_data_quality()

        self.contamination_level = -1
        self.max_combinations = max_combinations
        self.verbose = verbose

        # iqr_limit indicates how many multiples of the IQR below the 1st quartile or above the 3rd quartile are
        # considered outliers. This is used in numerous tests checking for small or large values. To reduce noise, a
        # stricter limit than the 1.5 or 2.2 normally used should be specified.
        self.iqr_limit = iqr_limit
        self.idr_limit = idr_limit

        # Synthetic data, if specified to create
        self.synth_df = None
        self.num_synth_rows = 1_000

        # The data to be examined and statistics related to it
        self.orig_df = None
        self.num_rows = -1
        self.column_medians = {}
        self.column_trimmed_means = {}
        self.column_unique_vals = {}  # Stored only for binary columns. Excludes None and NaN values.
        self.binary_cols = []
        self.numeric_cols = []
        self.date_cols = []
        self.string_cols = []

        # The number of tests executed. This will be the full set unless an list of tests to perform / excluded
        # is provided.
        self.n_tests_executed = 0
        self.execution_times = None

        # Information about the patterns found that have no exceptions
        self.patterns_arr = []
        self.patterns_df = None

        # A summary of the issues found, with a row for each test-feature describing the test and the exceptions.
        self.results_summary_arr = []
        self.results_summary_df = None

        # A dataframe indicating which cells in the original data were flagged by which tests. The rows correspond
        # to the rows in the original data. There is a column for each test-feature set where the pattern is true, but
        # there are exceptions. For example a test on columns A, B, and C would have, if exceptions were found, a
        # single column in test_results_df for that test and set of columns.
        self.test_results_df = None

        # This maps the columns used in test_results_df to the original columns
        self.col_to_original_cols_dict = {}

        # This maps one-to-one with the original data, but in each cell contains a score for the corresponding cell in
        # the original data. It has the same rows and columns as the original data. If a test finds an issue with
        # columns A, B, and C in rows 10 and 13, then test_results_by_column_np will have, in rows 10 & 13, in columns
        # A, B, and C, 0.33 each (in addition to any other issues these cells have been flagged for).
        self.test_results_by_column_np = None

        # Safe versions of the exceptions found. These are stored to support restore_issues()
        self.safe_results_summary_df = None
        self.safe_test_results_df = None
        self.safe_test_results_by_column_np = None

        # A dictionary describing each test. For each, we have the ID, description, method to test for the pattern
        # and exceptions, a method to generate synthetic data to demonstrate the test, and in indicator if the
        # pattern is in the patterns short list, the patterns listed by default in a call to get_patterns().
        self.test_dict = {

            # Tests on single columns of any type
            'MISSING_VALUES':           ('Check if all values are consistently present / consistently missing',
                                         self.__check_missing, self.__generate_missing,
                                         False, True, True),
            'RARE_VALUES':              ('Check if there are any rare values',
                                         self.__check_rare_values, self.__generate_rare_values,
                                         False, True, True),
            'UNIQUE_VALUES':            ('Check if there are consistently unique values',
                                         self.__check_unique_values, self.__generate_unique_values,
                                         True, True, True),
            'PREV_VALUES_DT':           (('Check if the values in a column can be predicted from previous values in '
                                         'that column'),
                                        self.__check_prev_values_dt, self.__generate_prev_values_dt,
                                        True, True, True),

            # Tests on pairs of columns of any type
            'MATCHED_MISSING':          ('Check if two columns have missing values consistently in the same rows',
                                         self.__check_matched_missing, self.__generate_matched_missing,
                                         True, True, False),
            'SAME_VALUES':              ('Check one column consistently has the same value as another column',
                                         self.__check_same, self.__generate_same,
                                         True, True, False),
            'SAME_OR_CONSTANT':         (('Check one column consistently has either the same value as another column, '
                                          'or a small number of other values'),
                                         self.__check_same_or_constant, self.__generate_same_or_constant,
                                         True, True, False),

            # Tests on single numeric columns
            'POSITIVE':                 ('Check if all numbers are positive',
                                         self.__check_positive_values, self.__generate_positive_values,
                                         False, True, True),
            'NEGATIVE':                 ('Check if all numbers are negative',
                                         self.__check_negative_values, self.__generate_negative_values,
                                         True, True, True),
            'NUMBER_DECIMALS':          (('Check if there is a consistent number of decimals in each value in the '
                                          'column'),
                                         self.__check_number_decimals, self.__generate_number_decimals,
                                         False, True, True),
            'COLUMN_ORDERED_ASC':       ('Check if the column is monotonically increasing',
                                         self.__check_column_increasing, self.__generate_column_increasing,
                                         True, True, True),
            'COLUMN_ORDERED_DESC':      ('Check if the column is monotonically decreasing',
                                         self.__check_column_decreasing, self.__generate_column_decreasing,
                                         True, True, True),
            'COLUMN_TENDS_ASC':         ('Check if the column is generally increasing',
                                         self.__check_column_tends_asc, self.__generate_column_tends_asc,
                                         True, True, True),
            'COLUMN_TENDS_DESC':        ('Check if the column is generally decreasing',
                                         self.__check_column_tends_desc, self.__generate_column_tends_desc,
                                         True, True, True),
            'SIMILAR_PREVIOUS':         ('Check if all values are similar to the previous value in the column, relative '
                                         'to the range of values in the column',
                                         self.__check_similar_previous, self.__generate_similar_previous,
                                         True, True, True),
            'UNUSUAL_NUMERIC':          (('Check if there are any unusual numeric values, in the sense of having an '
                                          'unusual order of magnitude'),
                                         self.__check_unusual_numeric, self.__generate_unusual_numeric,
                                         False, True, True),
            'FEW_NEIGHBORS':            (('Check if there are any unusual numeric values, in the sense of being '
                                          'distant from both the next smallest and next largest values'),
                                         self.__check_few_neighbors, self.__generate_few_neighbors,
                                         False, True, True),
            'FEW_WITHIN_RANGE':         (('Check if there are any unusual numeric values, in the sense of having few '
                                          'other values within a small range'),
                                         self.__check_few_within_range, self.__generate_few_within_range,
                                         False, True, True),
            'VERY_SMALL':               ('Check if there are any very small values',
                                         self.__check_very_small, self.__generate_very_small,
                                         True, True, True),
            'VERY_LARGE':               ('Check if there are any very large values',
                                         self.__check_very_large, self.__generate_very_large,
                                         True, True, True),
            'VERY_SMALL_ABS':           ('Check if there are any very small absolute values',
                                         self.__check_very_small_abs, self.__generate_very_small_abs,
                                         True, True, True),
            'MULTIPLE_OF_CONSTANT':     ('Check if all values are multiples of some constant',
                                         self.__check_multiple_constant, self.__generate_multiple_constant,
                                         True, True, True),
            'ROUNDING':                 ('Check if all values are rounded to the same degree',
                                         self.__check_rounding, self.__generate_rounding,
                                         True, True, True),
            'NON_ZERO':                 ('Check if all values are non-zero',
                                         self.__check_non_zero, self.__generate_non_zero,
                                         False, True, True),
            'LESS_THAN_ONE':            ('Check if all values are between -1.0 and 1.0, exclusive',
                                         self.__check_less_than_one, self.__generate_less_than_one,
                                         True, True, True),
            'GREATER_THAN_ONE':         ('Check if all values are less than -1.0 or greater than 1.0, inclusive',
                                         self.__check_greater_than_one, self.__generate_greater_than_one,
                                         False, True, True),

            # Tests on pairs of numeric columns
            'LARGER':                   ('Check if one column is consistently larger than another',
                                         self.__check_larger, self.__generate_larger,
                                         False, True, False),
            'MUCH_LARGER':              (('Check if one column is consistently at least 1 order of magnitude larger '
                                          'than another'),
                                         self.__check_much_larger, self.__generate_much_larger,
                                         False, True, False),
            'SIMILAR_WRT_RATIO':        (('Check if two columns are consistently similar, with respect to their ratio, '
                                          'to each other'),
                                         self.__check_similar_wrt_ratio, self.__generate_similar_wrt_ratio,
                                         True, True, False),
            'SIMILAR_WRT_DIFF':         (('Check if two columns are consistently similar, with respect to absolute '
                                          'difference, to each other'),
                                         self.__check_similar_wrt_difference, self.__generate_similar_wrt_difference,
                                         True, True, False),
            'SIMILAR_TO_INVERSE':       ('Check if one column is consistently similar to the inverse of the other',
                                         self.__check_similar_to_inverse, self.__generate_similar_to_inverse,
                                         True, True, False),
            'SIMILAR_TO_NEGATIVE':      ('Check if one column is consistently similar to the negative of the other',
                                         self.__check_similar_to_negative, self.__generate_similar_to_negative,
                                         True, True, False),
            'CONSTANT_SUM':             ('Check if the sum of two columns is consistently similar to a constant value',
                                         self.__check_constant_sum, self.__generate_constant_sum,
                                         True, True, False),
            'CONSTANT_DIFF':            (('Check if the difference of two columns is consistently similar to a constant'
                                          ' value'),
                                         self.__check_constant_diff, self.__generate_constant_diff,
                                         True, True, False),
            'CONSTANT_PRODUCT':         (('Check if the product of two columns is consistently similar to a constant '
                                          'value'),
                                         self.__check_constant_product, self.__generate_constant_product,
                                         True, True, False),
            'CONSTANT_RATIO':           (('Check if the ratio of two columns is consistently similar to a constant '
                                          'value'),
                                         self.__check_constant_ratio, self.__generate_constant_ratio,
                                         True, True, False),
            'EVEN_MULTIPLE':            ('Check if one column is consistently an even integer multiple of the other',
                                         self.__check_even_multiple, self.__generate_even_multiple,
                                         True, True, False),
            'RARE_COMBINATION':         ('Check if two columns have any unusual pairs of values',
                                         self.__check_rare_combination, self.__generate_rare_combination,
                                         True, True, False),
            'CORRELATED_FEATURES':      ('Check if two columns are consistently correlated',
                                         self.__check_correlated, self.__generate_correlated,
                                         True, True, False),
            'MATCHED_ZERO':             ('Check if two columns have a value of zero consistently in the same rows',
                                         self.__check_matched_zero, self.__generate_matched_zero,
                                         True, True, False),
            'OPPOSITE_ZERO':            (('Check if two columns are consistently such that one column contains a zero '
                                          'and the other contains a non-zero value'),
                                         self.__check_opposite_zero, self.__generate_opposite_zero,
                                         True, True, False),
            'RUNNING_SUM':              (('Check If one column is consistently the sum of its own value from the '
                                          'previous row and another column in the current row.'),
                                         self.__check_running_sum, self.__generate_running_sum,
                                         True, True, False),
            'A_ROUNDED_B':              ('Check If one column is consistently the result of rounding another column',
                                         self.__check_a_rounded_b, self.__generate_a_rounded_b,
                                         True, True, False),

            # Tests on pairs of columns where one must be numeric
            'MATCHED_ZERO_MISSING':     (('Check if two columns consistently have a zero in one and a missing value in '
                                         'the other'),
                                         self.__check_matched_zero_missing, self.__generate_matched_zero_missing,
                                         True, True, False),

            # Tests on sets of 3 numeric columns
            'SIMILAR_TO_DIFF':          (('Check if one column is consistently similar to the difference of two other '
                                          'columns'),
                                         self.__check_similar_to_diff, self.__generate_similar_to_diff,
                                         True, True, False),
            'DIFF_EXACT':               (('Check if one column is consistently exactly the difference of two other '
                                          'columns'),
                                         self.__check_diff_exact, self.__generate_diff_exact,
                                         True, False, False),
            'SIMILAR_TO_PRODUCT':       (('Check if one column is consistently similar to the product of two other '
                                         'columns'),
                                         self.__check_similar_to_product, self.__generate_similar_to_product,
                                         True, True, False),
            'PRODUCT_EXACT':            ('Check if one column is consistently exactly the product of two other columns',
                                         self.__check_product_exact, self.__generate_product_exact,
                                         True, False, False),
            'SIMILAR_TO_RATIO':         (('Check if one column is consistently similar to the ratio of two other '
                                          'columns'),
                                         self.__check_similar_to_ratio, self.__generate_similar_to_ratio,
                                         True, True, False),
            'RATIO_EXACT':              ('Check if one column is consistently exactly the ratio of two other columns',
                                         self.__check_ratio_exact, self.__generate_ratio_exact,
                                         True, False, False),
            'LARGER_THAN_SUM':          (('Check if one column is consistently larger than the sum of two other '
                                          'columns'),
                                         self.__check_larger_than_sum, self.__generate_larger_than_sum,
                                         False, True, False),
            'LARGER_THAN_ABS_DIFF':     (('Check if one column is consistently larger than the difference between two '
                                          'other columns'),
                                         self.__check_larger_than_abs_diff, self.__generate_larger_than_abs_diff,
                                         False, True, False),
            # todo: add LARGER_THAN_PRODUCT (not larger than ratio)
            # todo: Also SMALLER_THAN_ SUM / PRODUCT / ABS_DIFF

            # Tests on single numeric columns in relation to all other numeric columns
            'SUM_OF_COLUMNS':           (('Check if one column is consistently similar to the sum of two or more other '
                                          'columns'),
                                         self.__check_sum_of_columns, self.__generate_sum_of_columns,
                                         True, True, False),
            'MEAN_OF_COLUMNS':          (('Check if one column is consistently similar to the mean of two or more '
                                          'other columns'),
                                         self.__check_mean_of_columns, self.__generate_mean_of_columns,
                                         True, True, False),
            'MIN_OF_COLUMNS':           (('Check if one column is consistently similar to the sum of two or more other '
                                          'columns'),
                                         self.__check_min_of_columns, self.__generate_min_of_columns,
                                         True, True, False),
            'MAX_OF_COLUMNS':           (('Check if one column is consistently similar to the sum of two or more other '
                                          'columns'),
                                         self.__check_max_of_columns, self.__generate_max_of_columns,
                                         True, True, False),
            'ALL_POS_OR_ALL_NEG':       (('Identify sets of columns where the values are consistently either all '
                                         'positive, or all negative'),
                                         self.check_all_pos_or_all_neg, self.generate_all_pos_or_all_neg,
                                         True, True, False),
            'ALL_ZERO_OR_ALL_NON_ZERO': (('Identify sets of columns where the values are consistently either all '
                                          'zero, or non-zero'),
                                         self.check_all_zero_or_all_non_zero, self.generate_all_zero_or_all_non_zero,
                                         True, True, False),
            'DECISION_TREE_REGRESSOR':  (('Check if a numeric column can be derived from the other columns using a '
                                          'small decision tree'),
                                         self.__check_dt_regressor, self.__generate_dt_regressor,
                                         True, True, False),
            'LINEAR_REGRESSION':        (('Check if a numeric column can be derived from the other numeric columns '
                                          'using linear regression'),
                                         self.__check_lin_regressor, self.__generate_lin_regressor,
                                         True, True, False),
            'SMALL_VS_CORR_COLS':       (('Check if a value has an unusually small rank within its column compared to '
                                          'other ranks within that row for correlated columns'),
                                         self.__check_small_vs_corr_cols, self.__generate_small_vs_corr_cols,
                                         False, False, False),
            'LARGE_VS_CORR_COLS':       (('Check if a value has an unusually large rank within its column compared to '
                                          'other ranks within that row for correlated columns'),
                                         self.__check_large_vs_corr_cols, self.__generate_large_vs_corr_cols,
                                         False, False, False),

            # Tests on single Date columns
            'EARLY_DATES':              ('Check for dates significantly earlier than the other dates in the column',
                                         self.__check_early_dates, self.__generate_early_dates,
                                         True, True, True),
            'LATE_DATES':               ('Check for dates significantly later than the other dates in the column',
                                         self.__check_late_dates, self.__generate_late_dates,
                                         True, True, True),
            # todo: consider removing these tests, creatign a set of engineered featurs for each date giving its
            #   dow, dom, hour, and just checking for unusual values that way. Doing that would let us include
            #   those column in other tests. though it could get noisy too -- i'm not sure it's wise.
            'UNUSUAL_DAY_OF_WEEK':      ('Check if a date column contains any unusual days of the week',
                                         self.__check_unusual_dow, self.__generate_unusual_dow,
                                         True, True, True),
            'UNUSUAL_DAY_OF_MONTH':     ('Check if a date column contains any unusual days of the month',
                                         self.__check_unusual_dom, self.__generate_unusual_dom,
                                         True, True, True),
            'UNUSUAL_MONTH':            ('Check if a date column contains any months of the year',
                                         self.__check_unusual_month, self.__generate_unusual_month,
                                         True, True, True),
            'UNUSUAL_HOUR_OF_DAY':      (('Check if a datetime / time column contains any unusual hours of the day.'
                                          'This and UNUSUAL_MINUTES also identify where it is inconsistent if the time ' 
                                          'is included in the column.'),
                                         self.__check_unusual_hour, self.__generate_unusual_hour,
                                         True, True, True),
            'UNUSUAL_MINUTES':          ('Check if a datetime / time column contains any unusual minutes of the hour',
                                         self.__check_unusual_minutes, self.__generate_unusual_minutes,
                                         True, True, True),

            # Tests on pairs of date columns
            'CONSTANT_GAP':             (('Check if there is consistently a specific gap in time between two date '
                                          'columns'),
                                         self.__check_constant_date_gap, self.__generate_constant_date_gap,
                                         True, True, True),
            'LARGE_GAP':                ('Check if there is a larger than normal gap in time between two date columns',
                                         self.__check_large_date_gap, self.__generate_large_date_gap,
                                         True, True, True),
            'SMALL_GAP':                ('Check if there is a smaller than normal gap in time between two date columns',
                                         self.__check_small_date_gap, self.__generate_small_date_gap,
                                         True, True, True),
            'LATER':                    ('Check if one date column is consistently later than another date column',
                                         self.__check_date_later, self.__generate_date_later,
                                         True, True, True),

            # Tests on two columns, where one is date and the other is numeric
            'LARGE_GIVEN_DATE':         ('Check if a numeric value is very large given the value in a date column',
                                         self.__check_large_given_date, self.__generate_large_given_date,
                                         True, True, True),
            'SMALL_GIVEN_DATE':         ('Check if a numeric value is very small given the value in a date column',
                                         self.__check_small_given_date, self.__generate_small_given_date,
                                         True, True, True),

            # Tests on pairs of binary columns
            'BINARY_SAME':              (('For a pair of binary columns with the same set of 2 values, check if they '
                                          'consistently have the same value'),
                                         self.__check_binary_same, self.__generate_binary_same,
                                         True, True, False),
            'BINARY_OPPOSITE':           (('For a pair of binary columns with the same set of 2 values, check if they '
                                          'consistently have the opposite value'),
                                          self.__check_binary_opposite, self.__generate_binary_opposite,
                                          True, True, False),
            'BINARY_IMPLIES':           (('For a pair of binary columns with the same set of 2 values, check if '
                                          'when one has a given value, the other consistently does as well, though '
                                          'the other direction may not be true.'),
                                         self.__check_binary_implies, self.__generate_binary_implies,
                                         True, True, False),

            # Tests on sets of binary columns
            'BINARY_AND':               (('For sets of binary columns with the same set of 2 values, check if one'
                                          'column is consistently the result of ANDing the other columns'),
                                         self.__check_binary_and, self.__generate_binary_and,
                                         True, True, False),
            'BINARY_OR':                (('For sets of binary columns with the same set of 2 values, check if one '
                                          'column is consistently the result of ORing the other columns'),
                                         self.__check_binary_or, self.__generate_binary_or,
                                         True, True, False),
            'BINARY_XOR':               (('For sets of binary columns with the same set of 2 values, check if one '
                                          'column is consistently the result of XORing the other columns'),
                                         self.__check_binary_xor, self.__generate_binary_xor,
                                         True, True, False),
            'BINARY_NUM_SAME':         (('For sets of binary columns with the same set of 2 values, check if there is '
                                         'a consistent number of these columns with the same value'),
                                        self.__check_binary_num_same, self.__generate_binary_num_same,
                                        True, True, False),
            'BINARY_RARE_COMBINATION': ('Check for rare sets of values in sets of 3 or more binary columns',
                                        self.__check_binary_rare_combo, self.__generate_binary_rare_combo,
                                        True, True, False),

            # Tests on pairs of columns where one is binary and one is numeric
            'BINARY_MATCHES_VALUES':    (('Check if the binary column is consistently true when the values in a '
                                          'numeric column have low or when they have high values'),
                                         self.__check_binary_matches_values, self.__generate_binary_matches_values,
                                         True, True, False),

            # Tests on sets of three columns, where one must be binary
            'BINARY_TWO_OTHERS_MATCH':  (('Check if a binary column is consistently one value when two other columns '
                                         'have the same value as each other'),
                                         self.__check_binary_two_others_match, self.__generate_binary_two_others_match,
                                         True, True, False),
            # todo: add: BINARY_A_LARGER_B -> binary true/false consistently when A > B

            # Tests on sets of three columns, where one is binary and the other two string
            'BINARY_TWO_STR_SIMILAR':   (('Check if a binary column is consistently one value when two other string '
                                          'have similar values as each other, with respect to string length and '
                                          'the characters used'),
                                         self.__check_binary_two_str_match, self.__generate_binary_two_str_match,
                                         True, False, False),

            # Tests on sets of multiple columns, where one is binary and the others are numeric
            'BINARY_MATCHES_SUM':       (('Check if the binary column is consistently true when the sum of a set of '
                                         ' numeric columns is over some threshold'),
                                         self.__check_binary_matches_sum, self.__generate_binary_matches_sum,
                                         True, True, False),

            # Tests on single string columns
            'BLANK_VALUES':             ('Check for blank strings and values that are entirely whitespace',
                                         self.__check_blank, self.__generate_blank,
                                         False, True, True),
            'LEADING_WHITESPACE':       ('Check for strings with unusual leading whitespace',
                                         self.__check_leading_whitespace, self.__generate_leading_whitespace,
                                         True, True, True),
            'TRAILING_WHITESPACE':      ('Check for blank strings with unusual trailing whitespace',
                                         self.__check_trailing_whitespace, self.__generate_trailing_whitespace,
                                         True, True, True),
            'FIRST_CHAR_ALPHA':         ('Check if the first characters are consistently alphabetic',
                                         self.__check_first_char_alpha, self.__generate_first_char_alpha,
                                         True, True, True),
            'FIRST_CHAR_NUMERIC':       ('Check if the first characters are consistently numeric',
                                         self.__check_first_char_numeric, self.__generate_first_char_numeric,
                                         True, True, True),
            'FIRST_CHAR_SMALL_SET':     (('Check if there are a small number of distinct characters used for the first '
                                          'character'),
                                         self.__check_first_char_small_set, self.__generate_first_char_small_set,
                                         True, True, True),
            'FIRST_CHAR_UPPERCASE':     ('Check if the first character is consistently uppercase ',
                                         self.__check_first_char_uppercase, self.__generate_first_char_uppercase,
                                         True, True, True),
            'FIRST_CHAR_LOWERCASE':     ('Check if the first character is consistently lowercase ',
                                         self.__check_first_char_lowercase, self.__generate_first_char_lowercase,
                                         True, True, True),
            'LAST_CHAR_SMALL_SET':      (('Check if there are a small number of distinct characters used for the last '
                                          'character'),
                                         self.__check_last_char_small_set, self.__generate_last_char_small_set,
                                         True, True, True),
            'COMMON_SPECIAL_CHARS':     (('Check if there are one or more non-alphanumeric characters that '
                                          'consistently appear in the values'),
                                         self.__check_common_special_chars, self.__generate_common_special_chars,
                                         True, True, True),
            'COMMON_CHARS':             (('Check if there is consistently zero or a small number of characters '
                                          'repeated in the values'),
                                         # todo: look at renaming. see if COMMON_CHARS does make sense
                                         self.__check_count_repeated, self.__generate_count_repeated,
                                         True, True, True),
            'NUMBER_ALPHA_CHARS':       (('Check if there is a consistent number of alphabetic characters in each '
                                          'value in the column'),
                                         self.__check_number_alpha_chars, self.__generate_number_alpha_chars,
                                         True, True, True),
            'NUMBER_NUMERIC_CHARS':     (('Check if there is a consistent number of numeric characters in each value '
                                          'in the column'),
                                         self.__check_number_numeric_chars, self.__generate_number_numeric_chars,
                                         True, True, True),
            'NUMBER_ALPHANUMERIC_CHARS':(('Check if there is a consistent number of alphanumeric characters in  '
                                          'each value in the column '),
                                         self.__check_number_alphanumeric_chars,
                                         self.__generate_number_alphanumeric_chars,
                                         True, True, True),
            'NUMBER_NON-ALPHANUMERIC_CHARS': (('Check if there is a consistent number of non-alphanumeric characters '
                                              'in each value in the column '),
                                              self.__check_number_non_alphanumeric_chars,
                                              self.__generate_number_non_alphanumeric_chars,
                                              True, True, True),
            'NUMBER_CHARS':             (('Check if there is a consistent number of characters in each value in the '
                                          'column'),
                                         self.__check_number_chars, self.__generate_number_chars,
                                         True, True, True),
            'MANY_CHARS':               ('Check if any values have an unusually large number of characters ',
                                         self.__check_many_chars, self.__generate_many_chars,
                                         False, True, True),
            'FEW_CHARS':                ('Check if any values have an unusually small number of characters ',
                                         self.__check_few_chars, self.__generate_few_chars,
                                         False, True, True),
            'POSITION_NON-ALPHANUMERIC':('Check if the positions of the non-alphanumeric characters is consistent',
                                         self.__check_position_non_alphanumeric,
                                         self.__generate_position_non_alphanumeric,
                                         True, True, True),
            'UPPERCASE':                ('Check if all characters are consistently upper case ',
                                         self.__check_uppercase, self.__generate_uppercase,
                                         True, True, True),
            'LOWERCASE':                ('Check if all characters are consistently lower case ',
                                         self.__check_lowercase, self.__generate_lowercase,
                                         True, True, True),
            'CHARACTERS_USED':          (('Check if there is a consistent set of characters used in each value in the '
                                          'column'),
                                         self.__check_characters_used, self.__generate_characters_used,
                                         True, True, True),
            'FIRST_WORD_SMALL_SET':     ('Check if there is a small set of words consistently used for the first word',
                                         self.__check_first_word, self.__generate_first_word,
                                         True, True, True),
            'LAST_WORD_SMALL_SET':      ('Check if there is a small set of words consistently used for the last word',
                                         self.__check_last_word, self.__generate_last_word,
                                         True, True, True),
            'NUMBER_WORDS':             (('Check if there is a consistent number of words used in each value in the '
                                          'column'),
                                         self.__check_num_words, self.__generate_num_words,
                                         True, True, True),
            'LONGEST_WORDS':            ('Check if the column contains any unusually long words',
                                         self.__check_longest_words, self.__generate_longest_words,
                                         True, True, True),
            'COMMON_WORDS':             (('Check if there is a consistent set of words used in each value in the '
                                          'column'),
                                         self.__check_words_used, self.__generate_words_used,
                                         True, True, True),
            'RARE_WORDS':               ('Check if there are words which occur rarely in a given column ',
                                         self.__check_rare_words, self.__generate_rare_words,
                                         True, True, True),
            'GROUPED_STRINGS':          ('Check if a string or binary column is sorted into groups',
                                         self.__check_grouped_strings,
                                         self.__generate_grouped_strings,
                                         True, True, True),

            # Tests on pairs string columns
            'A_IMPLIES_B':              (('Check if specific values in one categorical column imply specific values in '
                                          'another categorical column'),
                                         self.__check_a_implies_b, self.__generate_a_implies_b,
                                         True, False, False),
            'RARE_PAIRS':               (('Checks for pairs of values in two columns, where neither is rare, but the '
                                          'combination is rare'),
                                         self.__check_rare_pairs, self.__generate_rare_pairs,
                                         True, True, False),
            'RARE_PAIRS_FIRST_CHAR':    (('Checks for pairs of values in two columns, where neither begins with a rare '
                                          'character, but the combination is rare'),
                                         self.__check_rare_pair_first_char, self.__generate_rare_pair_first_char,
                                         True, True, False),
            'RARE_PAIRS_FIRST_WORD':    (('Checks for pairs of values in two columns, where neither begins with a rare '
                                          'word, but the combination is rare'),
                                         self.__check_rare_pair_first_word, self.__generate_rare_pair_first_word,
                                         True, True, False),
            'RARE_PAIRS_FIRST_WORD_VAL':(('Checks for pairs of values in two columns, where the combination of the '
                                          'first word in one and the value in the other is rare'),
                                         self.__check_rare_pair_first_word_val,
                                         self.__generate_rare_pair_first_word_val,
                                         True, True, False),
            'SIMILAR_CHARACTERS':       (('Check if two string columns, with one word each, consistently have a '
                                          'significant overlap in the characters used.'),
                                         self.__check_similar_chars, self.__generate_similar_chars,
                                         True, True, False),
            'SIMILAR_NUM_CHARS':        ('Check if two string columns consistently have similar numbers of characters ',
                                         self.__check_similar_num_chars, self.__generate_similar_num_chars,
                                         True, True, False),
            'SIMILAR_WORDS':            (('Check if two string columns consistently have a significant overlap in the '
                                          'words used'),
                                         self.__check_similar_words, self.__generate_similar_words,
                                         True, True, False),
            'SIMILAR_NUM_WORDS':        ('Check if two string columns consistently have similar numbers of words ',
                                         self.__check_similar_num_words, self.__generate_similar_num_words,
                                         True, True, False),
            'SAME_FIRST_CHARS':         (('Check if two string columns consistently start with the same set of  '
                                          'characters.'),
                                         self.__check_same_first_chars, self.__generate_same_first_chars,
                                         True, True, False),
            'SAME_FIRST_WORD':         ('Check if two string columns consistently start with the same word.',
                                         self.__check_same_first_word, self.__generate_same_first_word,
                                         True, True, False),
            'SAME_LAST_WORD':          ('Check if two string columns consistently end with the same word.',
                                         self.__check_same_last_word, self.__generate_same_last_word,
                                         True, True, False),
            'SAME_ALPHA_CHARS':         (('Check if two string columns consistently contain the same set of  '
                                          'alphabetic characters.'),
                                         self.__check_same_alpha_chars, self.__generate_same_alpha_chars,
                                         True, True, False),
            'SAME_NUMERIC_CHARS':       (('Check if two string columns consistently contain the same set of  '
                                          'numeric characters.'),
                                         self.__check_same_numeric_chars, self.__generate_same_numeric_chars,
                                         True, True, False),
            'SAME_SPECIAL_CHARS':       (('Check if two string columns consistently contain the same set of  '
                                          'special characters.'),
                                         self.__check_same_special_chars, self.__generate_same_special_chars,
                                         True, True, False),
            'A_PREFIX_OF_B':            ('Check if one column is consistently the prefix of another column.',
                                         self.__check_a_prefix_of_b, self.__generate_a_prefix_of_b,
                                         True, True, False),
            'CORRELATED_ALPHA_ORDER':   ('Check if the alphabetic orderings of two columns are consistently correlated',
                                         self.__check_correlated_alpha, self.__generate_correlated_alpha,
                                         True, True, False),

            # Tests with one string and one numeric column
            'LARGE_GIVEN_VALUE':        (('Check if the value in the numeric column is very large given the value in '
                                          'the categorical column'),
                                         self.__check_very_large_given, self.__generate_very_large_given,
                                         True, True, False),
            'SMALL_GIVEN_VALUE':        (('Check if the value in the numeric column is very small given the value in '
                                          'the categorical column'),
                                         self.__check_very_small_given, self.__generate_very_small_given,
                                         True, True, False),
            'LARGE_GIVEN_PREFIX':       (('Check if the value in the numeric column is very large given the first '
                                          'word in the categorical column'),
                                          self.__check_large_given_prefix, self.__generate_large_given_prefix,
                                          True, True, False),
            'SMALL_GIVEN_PREFIX':       (('Check if the value in the numeric column is very small given the first '
                                          'word in the categorical column'),
                                          self.__check_small_given_prefix, self.__generate_small_given_prefix,
                                          True, True, False),
            'GROUPED_STRINGS_BY_NUMERIC':(('Check if a string or binary column is sorted into groups when the table is'
                                           'ordered by a numeric or date column'),
                                          self.__check_grouped_strings_by_numeric,
                                          self.__generate_grouped_strings_by_numeric,
                                          True, True, False),

            # Tests on two string and one numeric column
            'LARGE_GIVEN_PAIR':         (('Check if a numeric or date column is large given the pair of values in two '
                                          'string or binary columns'),
                                         self.__check_large_given_pair, self.__generate_large_given_pair,
                                         True, True, False),
            'SMALL_GIVEN_PAIR':         (('Check if a numeric or date column is large given the pair of values in two '
                                          'string or binary columns'),
                                         self.__check_small_given_pair, self.__generate_small_given_pair,
                                         True, True, False),

            # Tests on one string/binary column and two numeric
            'CORRELATED_GIVEN_VALUE':   (('Check if two numeric columns are correlated if conditioning on a string or'
                                         'binary column'),
                                         self.__check_corr_given_val, self.__generate_corr_given_val,
                                         True, True, False),

            # Tests on one string column related to the rest of the columns
            'DECISION_TREE_CLASSIFIER': (('Check if the categorical column can be derived from the other columns using '
                                          'a decision tree'),
                                         self.__check_dt_classifier, self.__generate_dt_classifier,
                                         True, True, False),

            # Tests on sets of four columns of any type
            'TWO_PAIRS':                ('Check that, given two pairs of columns, the first pair have matching values '
                                         'in the same rows as the other pair',
                                         self.__check_two_pairs, self.__generate_two_pairs,
                                         True, True, False),

            # Tests on sets of columns of any type
            'UNIQUE_SETS_VALUES':       ('Check if a set of columns has unique combinations of values ',
                                         self.__check_unique_sets_values, self.__generate_unique_sets_values,
                                         True, True, False),

            # Tests on rows of values
            'MISSING_VALUES_PER_ROW':   ('Check if there is a consistent number of missing values per row',
                                         self.__check_missing_values_per_row, self.__generate_missing_values_per_row,
                                         True, True, True),
            'ZERO_VALUES_PER_ROW':      ('Check if there is a consistent number of zero values per row',
                                         self.__check_zero_values_per_row, self.__generate_zero_values_per_row,
                                         True, True, True),
            'UNIQUE_VALUES_PER_ROW':    ('Check if there is a consistent number of unique values per row',
                                         self.__check_unique_values_per_row, self.__generate_unique_values_per_row,
                                         True, True, True),
            'NEGATIVE_VALUES_PER_ROW':  ('Check if there is a consistent number of negative values per row',
                                         self.__check_negative_values_per_row, self.__generate_negative_values_per_row,
                                         True, True, True),
            'SMALL_AVG_RANK_PER_ROW':   (('Check if the numeric values in a row have a small average percentile value '
                                          'relative to their columns. This indicates the numeric values are typically'
                                          'unusually small.'),
                                         self.__check_small_avg_rank_per_row, self.__generate_small_avg_rank_per_row,
                                         False, True, True),
            'LARGE_AVG_RANK_PER_ROW':   (('Check if the numeric values in a row have a large average percentile value '
                                          'relative to their columns. This indicates the numeric values are typically'
                                          'unusually large.'),
                                         self.__check_large_avg_rank_per_row, self.__generate_large_avg_rank_per_row,
                                         False, True, True),
        }

        # Check if the specified tests are valid, now that the set of valid tests are defined.
        specified_test_list = []
        if exclude_list:
            specified_test_list.extend(exclude_list)
        if execute_list:
            specified_test_list.extend(execute_list)
        for t in specified_test_list:
            if t not in self.test_dict.keys():
                print(f"Error {t} is not a valid test")

    def __prepare_data(self, df, date_cols):
        self.orig_df = df

        # Ensure the dataframe has at least a very minimum number of rows.
        if len(df) < 10:
            print(f"The dataframe contains too few rows to be examined in a meaningful manner. Number of rows: "
                  f"{len(df)}")
            return None

        # Ensure the dataframe has a predictable index, which may be kept inline with parallel dataframes created
        # running the tests.
        self.orig_df = self.orig_df.reset_index(drop=True)
        self.num_rows = len(self.orig_df)
        self.num_valid_rows = None

        # Ensure the dataframe has column names in string format. Often dataframes contain numeric column names.
        self.orig_df.columns = [str(x) for x in self.orig_df.columns]

        # Remove any columns where there are not two or more unique values, or the values are all Null
        cols = [x for x in self.orig_df.columns if
                (self.orig_df[x].nunique(dropna=False) > 1) and (self.orig_df[x].isna().sum() < self.num_rows)]
        if self.verbose >= 2 and len(cols) < len(self.orig_df.columns):
            removed_cols = set(self.orig_df.columns) - set(cols)
            print()
            print(f"Removing columns with only one value: {removed_cols}")
        self.orig_df = self.orig_df[cols]

        # Determine which columns are binary, numeric, date, and neither. Any that are not binary, numeric, or date
        # we consider as string.
        self.binary_cols = []
        self.numeric_cols = []
        self.date_cols = []
        self.string_cols = []
        for col_name in self.orig_df.columns:
            if self.orig_df[col_name].nunique() == 2:
                self.binary_cols.append(col_name)
            elif pandas_types.is_numeric_dtype(self.orig_df[col_name]) or \
                    self.orig_df[col_name].astype(str).str.isdigit().tolist().count(False) < self.contamination_level:
                self.numeric_cols.append(col_name)
            elif self.orig_df[col_name].dtype in [np.datetime64, 'datetime64[ns]']:
                self.date_cols.append(col_name)
            else:
                try:
                    _ = self.orig_df[col_name].astype(float)
                    self.numeric_cols.append(col_name)
                except Exception:
                    self.string_cols.append(col_name)

        # Try to convert any string columns we can to date format. The code below is likely sufficient, though may
        # erroneously convert some string or numeric columns to dates. If we find any legitimate date columns are
        # missed, we can use PyTime: https://github.com/shinux/PyTime. We are trying to minimize the pip installs
        # necessary for this tool, so will only add this if necessary.

        if date_cols is None:
            new_date_cols = []
            for col_name in self.string_cols:
                # Do not convert to date if the strings are too short. They must be at least yyyymm (6 characters)
                if statistics.median(self.orig_df[col_name].astype(str).str.len()) < 6:
                    continue
                try:
                    df[col_name] = pd.to_datetime(self.orig_df[col_name])
                    new_date_cols.append(col_name)
                    self.date_cols.append(col_name)
                except Exception:
                    pass
            for datecol in new_date_cols:
                self.string_cols.remove(datecol)
        else:
            new_date_cols = []
            for col_name in date_cols:
                try:
                    df[col_name] = pd.to_datetime(self.orig_df[col_name])
                    new_date_cols.append(col_name)
                    self.date_cols.append(col_name)
                except Exception:
                    pass
            for datecol in new_date_cols:
                self.string_cols.remove(datecol)

        # todo: for columns with mostly numeric values, but some non-alphanumeric (eg $, %), strip and do the numeric tests
        #   maybe do this in the prepare_data(), duplicate the columns.

        # For binary columns, find and cache the set of unique values per column
        for col_name in self.binary_cols:
            self.column_unique_vals[col_name] = sorted([x for x in self.orig_df[col_name].unique() if not is_missing(x)])

        # Calculate and cache the pairwise correlations between each numeric column
        self.pearson_corr = self.orig_df.corr(method='pearson')
        self.spearman_corr = self.orig_df.corr(method='spearman')

        # Calculate and cache the median for each numeric column.
        self.column_medians = {}
        for col_name in self.numeric_cols:
            self.column_medians[col_name] = self.orig_df[col_name].median()

        # Calculate the trimmed mean for each numeric column
        trimmed_orig_df = self.orig_df.copy()
        for col_name in self.numeric_cols:
            lower_limit = self.orig_df[col_name].quantile(0.01)
            upper_limit = self.orig_df[col_name].quantile(0.99)
            trimmed_orig_df = trimmed_orig_df[(trimmed_orig_df[col_name] > lower_limit) &
                                              (trimmed_orig_df[col_name] < upper_limit)]
        self.column_trimmed_means = {}
        for col_name in self.numeric_cols:
            self.column_trimmed_means[col_name] = trimmed_orig_df[col_name].mean()

        # Create a sample of the full data, which may be used for early stopping for expensive tests.
        # The sample_df will tend to not contain Null values.
        if len(trimmed_orig_df) > 50:
            self.sample_df = trimmed_orig_df.dropna().sample(n=min(len(trimmed_orig_df.dropna()), 50))
        elif len(self.orig_df.dropna()) > 50:
            self.sample_df = self.orig_df.dropna().sample(n=50)
        else:
            self.sample_df = self.orig_df.sample(n=min(len(self.orig_df), 50))

        # Calculate the contamination rate on the sample. This is the maximum number of rows in the sample which may be
        # in violation of a pattern where we still consider the pattern to potentially hold.
        num_sample_rows = len(self.sample_df)
        contamination_frac = self.contamination_level / self.num_rows

        # Calculate the number of valid (ie, not missing) values there are per row
        self.num_valid_rows = {}
        for col_name in self.orig_df.columns:
            self.num_valid_rows[col_name] = len([x for x in self.orig_df[col_name] if not is_missing(x)])

        # patterns_df has a row for each test for each feature where there is a pattern with no exceptions.
        self.patterns_arr = []
        self.patterns_df = None

        # test_results_df has a column for each test for each original column where there is a pattern and also
        # exceptions.
        self.test_results_df = pd.DataFrame()
        self.n_tests_executed = 0

        # test_results_by_column_np is set initially to all zeros, as no
        self.test_results_by_column_np = np.zeros((self.num_rows, len(self.orig_df.columns)), dtype=float)

        # results_summary_df has row for each test for each feature where there is a pattern and also exceptions.
        self.results_summary_arr = []
        self.results_summary_df = None

        if self.verbose >= 2:
            print()
            print("Identified column types:")
            print(f"Number string: {len(self.string_cols)}")
            print(f"Number numeric: {len(self.numeric_cols)}")
            print(f"Number date/time: {len(self.date_cols)}")
            print(f"Number binary: {len(self.binary_cols)}")
            print()

    def generate_synth_data(self, all_cols=False, seed=0, add_nones="none"):
        """
        Generate a random synthetic dataset which may be used to demonstrate each of the tests.
        If all_cols is False, this generates columns specifically only for the tests that are specified to run.
        If all_cols is True, this generates columns related to all tests, even where that test is not specified to run.

        If add_nones is set to 'random', then each column will have a set of None values added randomly, covering 50%
        the values.
        If set to 'in-sync', this is similar, but all columns will have None set in the ame rows.
        """
        assert add_nones in ['none', 'one-row', 'in-sync', 'random']

        random.seed(seed)
        self.synth_df = pd.DataFrame()
        for test_id in self.test_dict.keys():
            if all_cols or \
                    ((self.execute_list is None and self.exclude_list is None) or
                     (self.execute_list and test_id in self.execute_list) or
                     (self.exclude_list and test_id not in self.exclude_list)):
                self.test_dict[test_id][TEST_DEFN_GEN_FUNC]()

        if add_nones == 'one-row':
            # Set a single row, all columns to None. This checks that the tests are able to handle at least some Nulls
            for col_name in self.synth_df.columns:
                self.synth_df.loc[0, col_name] = None
        elif add_nones == 'in-sync':
            # Do not set the last few rows, where the exceptions tend to be, as None
            none_idxs = random.sample(range(self.num_synth_rows - 10), self.num_synth_rows // 2)
            for col_name in self.synth_df.columns:
                col_vals = self.synth_df[col_name].copy()
                col_vals.iloc[none_idxs] = None
                self.synth_df[col_name] = col_vals
        elif add_nones == 'random':
            for col_name in self.synth_df.columns:
                none_idxs = random.sample(range(self.num_synth_rows - 10), self.num_synth_rows // 2)
                col_vals = self.synth_df[col_name].copy()
                col_vals.iloc[none_idxs] = None
                self.synth_df[col_name] = col_vals

        return self.synth_df

    def check_data_quality(
            self,
            df,
            date_cols=None,
            test_start_id=0,
            fast_only=False,
            contamination_level=0.005,
            run_parallel=False):
        """
        Run the specified tests on the dataset specified here. The tests are specified in the constructor. This method
        populates the patterns, results_summary, and results dataframes.

        Args:
            df: the dataframe containing all data to be assessed

            date_cols: If specified, these, and only these, columns will be treated as date columns.

            test_start_id: Each test has a unique number. Specifying a value greater than 0 will skip the initial tests.
                This may be specified to continue a previous execution that was incomplete.

            fast_only: If specified, only tests that operate on single columns will be executed. The slower tests check
                sets of two or more columns, and are skipped if this is set True.

            contamination_level: The maximum fraction of rows in violation of the pattern where we consider the pattern
                to still be in place. If set as an integer, this defines the maximum number of rows, as opposed to the
                fraction.

            run_parallel: fill in!!!!!!!

        Returns:
            This returns the results_summary dataframe.
        """

        if df is None or len(df) == 0:
            return None

        # Check the dataframe does not contain duplicate column names. If it does, rename all columns with a _idx at
        # the end of each
        if len(df.columns) > len(set(df.columns)):
            df = df.copy()
            print(("Duplicate column names encountered. A suffix will be added to all column names to ensure they are "
                   f"unique. Duplicate column names: {set([x for x in df.columns if df.columns.tolist().count(x) > 1])}"))
            new_col_names = []
            for col_idx, col_name in enumerate(df.columns):
                new_col_names.append(f"{col_name}_{col_idx}")
            df.columns = new_col_names

        # Store the contamination_level in terms of number of rows. It may have been passed either in this form or as a
        # fraction.
        if contamination_level > 1 and type(contamination_level) == int:
            if contamination_level > len(df):
                print((f"Error. contamination rate set to {contamination_level}, more than the number of rows in the "
                       f"dataframe passed. The contamination_level rate should be substantially smaller."))
                return None
            self.contamination_level = contamination_level
        elif contamination_level < 1.0:
            self.contamination_level = contamination_level * len(df)
            if self.contamination_level < 1.0:
                if contamination_level != 0.005:
                    print((f"Error. contamination rate set to {contamination_level}, not allowing even 1 row to be in "
                           "violation of the patterns. Must be set to a larger value given the number of rows "
                           "available."))
                    return None
                self.contamination_level = 1

        self.__prepare_data(df, date_cols)

        # Adjust the test_start_id to 0 if necessary. 0 is the lowest valid value.
        if test_start_id < 0:
            test_start_id = 0

        # Determine the set of tests to execute
        self.execution_test_list = []
        for test_idx, test_id in enumerate(self.test_dict.keys()):
            if test_idx < test_start_id:
                continue
            if fast_only and not self.test_dict[test_id][TEST_DEFN_FAST]:
                continue
            if (self.execute_list is None and self.exclude_list is None) or \
                    (self.execute_list and test_id in self.execute_list) or \
                    (self.exclude_list and test_id not in self.exclude_list):
                if self.test_dict[test_id][TEST_DEFN_IMPLEMENTED]:
                    self.execution_test_list.append(test_id)

        # Check at least one valid test was specified
        if len(self.execution_test_list) == 0:
            print("No valid tests specified.")
            return None

        # Execute all tests
        self.n_tests_executed = 0
        self.execution_times = {}

        # Get the test index of each test
        test_idx_dict = {x:y for x, y in
                         zip(self.test_dict.keys(), range(len(self.test_dict.keys())))
                         if self.test_dict[x][TEST_DEFN_IMPLEMENTED]}

        if run_parallel:
            process_arr = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for test_id in self.execution_test_list:
                    if self.verbose:
                        self.__output_current_test(test_idx_dict[test_id], test_id)
                    #f = executor.submit(self.test_dict[test_id][TEST_DEFN_FUNC], test_id)
                    #func = self.test_dict[test_id][TEST_DEFN_FUNC]
                    f = executor.submit(call_test, self, test_id)
                    process_arr.append(f)
                    self.n_tests_executed += 1
                for f in process_arr:
                    f.result()
        else:
            for test_id in self.execution_test_list:
                if self.verbose:
                    self.__output_current_test(test_idx_dict[test_id], test_id)
                try:
                    t1 = time.time()
                    self.test_dict[test_id][TEST_DEFN_FUNC](test_id=test_id)
                    t2 = time.time()
                    self.execution_times[test_id] = t2 - t1
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    line_number_str = str(exc_tb.tb_lineno)
                    while exc_tb.tb_next:
                        exc_tb = exc_tb.tb_next
                        line_number_str += " -- " + str(exc_tb.tb_lineno)
                    print(colored(f"Error executing {test_id}: {e}, line number: {line_number_str}", 'red'))
                self.n_tests_executed += 1

        # Sum the the number of columns flagged per row
        self.__calculate_final_scores()

        # Create the final dataframes of patterns and exceptions found
        self.patterns_df = pd.DataFrame(
            self.patterns_arr,
            columns=['Test ID', 'Column(s)', 'Description of Pattern', 'Display Information'])
        self.results_summary_df = pd.DataFrame(
            self.results_summary_arr,
            columns=['Test ID', 'Column(s)', 'Description of Pattern', 'Number of Exceptions', 'Display Information'])

        # Save safe versions of the exceptions found, to support restore_issues() if called
        self.safe_results_summary_df = self.results_summary_df.copy()
        self.safe_test_results_df = self.test_results_df.copy()
        self.safe_test_results_by_column_np = self.test_results_by_column_np.copy()

        # Display summary statistics
        self.__output_stats()

        return self.get_results_summary()

    def get_test_list(self):
        return [x for x in self.test_dict.keys() if self.test_dict[x][TEST_DEFN_IMPLEMENTED]]

    def get_test_descriptions(self):
        return {x: self.test_dict[x][TEST_DEFN_DESC] for x in self.test_dict.keys() if self.test_dict[x][TEST_DEFN_IMPLEMENTED]}

    def print_test_descriptions(self, long_desc=False):
        for test_id in self.test_dict.keys():
            text = self.test_dict[test_id][TEST_DEFN_DESC]
            if long_desc:
                doc_str = self.test_dict[test_id][TEST_DEFN_FUNC].__doc__
                if doc_str:
                    text += doc_str
                    text = ' '.join(text.split())
            multiline_test_desc = wrap(text, 90)
            print(f"{test_id+':':<30} {multiline_test_desc[0]}")
            filler = ''.join([" "]*32)
            for line in multiline_test_desc[1:]:
                print(f'{filler} {line}')

    #########################################################################
    # Output the results of the analysis in various ways
    #########################################################################

    def get_patterns(self, test_exclude_list=None, column_exclude_list=None, short_list=True):
        # todo: return the patterns in a denser format, or provide an option to do so. Some tests possibly don't
        #   present by default, such as POSITIVE, LARGER, MUCH LARGER, etc. Just indicate not showing and explain
        #   how to print if desired.
        #   If we do display, for example, LARGER, then instead of listing every pair, for each column, list
        #   the other columns it is larger than.

        if self.patterns_df is None:
            return None

        if test_exclude_list is None and column_exclude_list is None and short_list == False:
            return self.patterns_df.drop(columns=['Display Information'])
        df = self.patterns_df.copy()
        if test_exclude_list:
            df = df[~df['Test ID'].isin(test_exclude_list)]
        if column_exclude_list:
            df = df[~df['Column(s)'].isin(column_exclude_list)]
        if short_list:
            df = df[df['Test ID'].isin(self.get_patterns_shortlist())]
        return df.drop(columns=['Display Information'])

    def get_patterns_shortlist(self):
        # The 3rd element for each item in test_dict indicates if that test is in the short-list for reporting
        # discovered patterns
        return [x for x in self.test_dict.keys() if self.test_dict[x][TEST_DEFN_SHORTLIST]]

    def get_results_summary(self):
        return self.results_summary_df.drop(columns=['Display Information'])

    def get_test_results(self):
        return self.test_results_df

    def get_test_results_by_column(self):
        return pd.DataFrame(self.test_results_by_column_np, columns=self.orig_df.columns)

    def summarize_patterns_by_test_and_feature(self, all_tests=False, heatmap=False):
        """
        Create and return a dataframe with a row for each test and a column for each feature in the original data. Each
        cell has a 0 or 1, indicating if the pattern was found without exceptions in that feature. Note, some tests to
        not identify patterns, such as VERY_LARGE. The dataframe is returned, and optionally displayed as a heatmap.

        all_tests: If all_tests is True, all tests are included in the output, even those that found no patterns. This
        may be used specifically to check which found no patterns.

        heatmap: If True, a heatmap will be displayed
        """

        if self.patterns_df is None:
            return None

        summary_arr = []
        cols = ['Test ID']
        cols.extend(self.orig_df.columns)
        for test_id in self.get_test_list():
            test_arr = [test_id]
            test_sub_df = self.patterns_df[self.patterns_df['Test ID'] == test_id]
            if (not all_tests) and (len(test_sub_df) == 0):
                continue
            for col_name in self.orig_df.columns:
                if col_name in test_sub_df['Column(s)'].tolist():
                    test_arr.append(1)
                else:
                    test_arr.append(0)
            # todo: this is wrong where there are multiple tests and they flag different rows. Without this though
            # we skip all patterns over 2, 3+ columns, so need to finish this.
            # for col_name in test_sub_df['Column(s)'].tolist():
            #     if col_name not in self.orig_df.columns:
            #         cols.append(col_name)
            #         test_arr.append(1)
            summary_arr.append(test_arr)
        df = pd.DataFrame(summary_arr, columns=cols)
        df = df.set_index("Test ID", drop=True)
        if heatmap and len(df):
            plt.subplots(figsize=(len(df.columns) * 0.5, len(df) * 0.5))
            s = sns.heatmap(df, cmap='Blues', linewidths=1.1, linecolor='black', annot=True, fmt="d")
            s.set(ylabel=None)
            plt.show()

        return df

    def summarize_results_by_test(self, heatmap=False):
        """
        Create and return a dataframe with a row for each test, indicating 1) the number of features where the pattern
        was found but also exceptions, 2) the number of issues total flagged (across all features and all rows).

        heatmap: if True, a heatmap of the results are displayed
        """

        if self.results_summary_df is None:
            return None

        # todo: add 'Number of Rows Flagged At Least Once'
        g = self.results_summary_df.groupby('Test ID')
        df = pd.DataFrame({
            'Test ID': list(g.groups),
            'Number of Columns Flagged At Least Once': list(g['Column(s)'].nunique()),
            'Number of Issues Total': list(g['Number of Exceptions'].sum())
        })

        # Use the consistent ordering for the tests
        df['Test ID'] = df['Test ID'].astype("category")
        df['Test ID'] = df['Test ID'].cat.set_categories(self.get_test_list())
        df = df.sort_values(['Test ID'])

        df = df.set_index("Test ID", drop=True)
        if heatmap:
            fig, ax = plt.subplots(figsize=(len(df.columns) * 0.5, len(df) * 0.5))
            s = sns.heatmap(df, cmap='Blues', linewidths=0.75, linecolor='black', annot=True, fmt="d")
            s.set(ylabel=None)
            s.set(xticklabels=[])
            print("Counts: Number Columns Flagged at Least Once, and Total Number of Issues:")
            plt.show()
        return df

    # todo: also have a method to summarize by feature, but first finalize how we save features for tests based on 2/3 features

    def summarize_results_by_test_and_feature(self, all_tests=False, heatmap=False):
        """
        Create a dataframe with a row for each test and a column for each feature in the original data. Each cell has an
        integer, indicating, if the pattern was found in that feature, the number of rows that were flagged. Note, some
        # at most 0.5% of the rows may be flagged for any test in any feature, as this checks for exceptions to well-
        established patterns.

        If all_tests is True, all tests are included in the output, even those that found no issues. This may be
        used specifically to check which found no issues.
        """

        if self.results_summary_df is None:
            return None

        # todo: this misses where a test checks 2 or 3 columns.
        summary_arr = []
        for test_id in self.get_test_list():
            test_arr = [test_id]
            test_sub_df = self.results_summary_df[self.results_summary_df['Test ID'] == test_id]
            if (not all_tests) and (len(test_sub_df) == 0):
                continue
            for col_name in self.orig_df.columns:
                result_col_name = self.get_results_col_name(test_id, col_name)
                if result_col_name in self.test_results_df.columns:
                    v = self.test_results_df[result_col_name].tolist().count(True)
                    test_arr.append(v)
                else:
                    test_arr.append(0)
            summary_arr.append(test_arr)
        cols = ['Test ID']
        cols.extend(self.orig_df.columns)
        df = pd.DataFrame(summary_arr, columns=cols)

        df = df.set_index("Test ID", drop=True)
        if heatmap:
            plt.subplots(figsize=(len(df.columns)*0.5, len(df) * 0.5))
            s = sns.heatmap(df, cmap='Blues', linewidths=1.1, linecolor='black', annot=True, fmt="d")
            s.set(ylabel=None)
            plt.show()

        return df

    def summarize_patterns_and_results(self, heatmap=False):
        """
        This indicates which tests found a pattern on which columns, regardless if there were exceptions.
        """
        # todo: use white for no pattern, blue for pattern with no exceptions, yellow for pattern with exceptions.
        pass

    def __display_rows_with_tests(self, sorted_df, nrows, check_score=False):
        """
        Display a set of dataframes, one per row in the original data, up to nrows rows, each including the original row
        and the issues found in it, across all tests on all features.

        sorted_df: a sorted version of self.test_results_df
        nrows: the maximum number of rows from the original data to display
        check_score: if True, only rows with scores above zero will be displayed
        """

        # todo: this misses where a test checks 2 or 3 columns. -- need to flag all columns
        flagged_idx_arr = sorted_df.index[:10]
        for row_idx in flagged_idx_arr[:nrows]:
            if check_score and sorted_df.loc[row_idx]['FINAL SCORE'] == 0:
                print(f"The remaining rows have no flagged issues: cannot display {nrows} flagged rows.")
                return

            # Get the row as it appears in the original data
            orig_row = self.orig_df.loc[row_idx:row_idx]

            # Insert a column to indicate the IDs of the tests that have flagged this row
            orig_row.insert(0, 'Test ID', '')

            colour_cells = [False] * len(self.orig_df.columns)

            # Loop through all tests, and add a row to the output for any that have flagged this row
            for test_id in self.get_test_list():
                test_row = [test_id] + [""] * len(self.orig_df.columns)

                # There may be multiple columns / column sets which have flagged this row.
                for column_set in self.results_summary_df['Column(s)'].unique():
                    result_col_name = self.get_results_col_name(test_id, column_set)
                    if result_col_name not in self.test_results_df.columns:
                        continue
                    for column_name in self.col_to_original_cols_dict[result_col_name]:
                        if self.test_results_df[result_col_name][row_idx]:
                            column_idx = np.where(self.orig_df.columns == column_name)[0][0]
                            test_row[column_idx+1] = "Flagged"
                            colour_cells[column_idx] = True
                if test_row.count("Flagged"):
                    orig_row = orig_row.append(pd.DataFrame(test_row, index=orig_row.columns).T)
            orig_row = orig_row.reset_index()
            orig_row = orig_row.drop(columns=['index'])

            # Display the dataframe representing this row from the original data
            print()
            if is_notebook():
                display(Markdown(f"**Row: {row_idx}  --  Final Score: {sorted_df.loc[row_idx]['FINAL SCORE']}**"))
                display(orig_row.style.apply(styling_orig_row, row_idx=0, flagged_arr=colour_cells, axis=None))
            else:
                print(f"Row: {row_idx} Final Score: {sorted_df.loc[row_idx]['FINAL SCORE']}")
                print(orig_row.to_string(index=False))
            print()

    def display_least_flagged_rows(self, with_results=True, nrows=10):
        """
        This displays the nrows rows from the original data with the lowest scores. These are the rows with the least
        flagged issues.

        with_results: If with_results is False, this displays a single dataframe showing the appropriate subset
        of the original data. If with_results is True, this displays a dataframe per original row, up to nrows. For
        each, the original data is shown, along with all flagged issues, across all tests on all features.

        nrows: the maximum number of original rows to present.
        """
        sorted_df = self.test_results_df.sort_values('FINAL SCORE', ascending=True)
        if with_results:
            self.__display_rows_with_tests(sorted_df, nrows)
        else:
            df = self.orig_df.loc[sorted_df.index].head(nrows).copy()
            df['FINAL SCORE'] = sorted_df['FINAL SCORE'][:len(df)]
            if is_notebook():
                display(df)
            else:
                print(df)

    def display_most_flagged_rows(self, with_results=True, nrows=10):
        """
        This is similar to display_least_flagged_rows, but displays the rows with the most identified issues.
        """

        if self.test_results_df is None or len(self.test_results_df) == 0:
            return None

        sorted_df = self.test_results_df.sort_values('FINAL SCORE', ascending=False)
        sorted_df.index = [x[0] if type(x) == tuple else x for x in sorted_df.index]
        if with_results:
            self.__display_rows_with_tests(sorted_df, nrows, check_score=True)
        else:
            row_idxs = np.array(sorted_df.index.to_list()).reshape(1, -1)[0]
            df = self.orig_df.loc[row_idxs].head(nrows).copy()
            df['FINAL SCORE'] = sorted_df['FINAL SCORE'][:len(df)]
            df = df[df['FINAL SCORE'] > 0]
            if is_notebook():
                display(df.style.apply(styling_flagged_rows,
                                       flagged_cells=self.test_results_by_column_np,
                                       flagged_arr=[True] * len(self.orig_df.columns), axis=None))
            else:
                print(df)

    def get_rows_flagged(self, test_id, col_name):
        """
        Return the subset of the original data where the specified test flagged an issue in the specified column
        """
        results_col_name = self.get_results_col_name(test_id, col_name)
        if results_col_name not in self.test_results_df.columns:
            return None
        df = self.test_results_df[self.test_results_df[results_col_name] == 1]
        if len(df) == 0:
            return None
        df.index = [x[0] if type(x) == tuple else x for x in df.index]
        row_idxs = df.index
        return self.orig_df.loc[row_idxs]

    def get_sample_not_flagged(self, test_id, col_name, n_examples=10, group_results=False, is_patterns=False):
        """
        Return a random set of n_examples values from the specified column that were not flagged by the
        specified test in the specified columns.

        test_id: If test_id is None, this returns values not flagged by any test.
        col_name: single column or column set
        n_examples: the maximum number of examples to return.
        group_results: if True, this will return consecutive rows. This used for tests where row order is relevant.
        is_patterns: True if this is being called to display examples of a pattern, in which case there are no
            exceptions.
        """
        if test_id is not None:
            if is_patterns:
                cols = [x.lstrip('"').rstrip('"') for x in col_name.split(" AND ")]
            else:
                results_col_name = self.get_results_col_name(test_id, col_name)
                cols = self.col_to_original_cols_dict[results_col_name]

            # If there are no values flagged for this test in this feature, there will not be a column in
            # test_results_df. In this case, return any values.
            if not is_patterns and results_col_name not in self.test_results_df.columns:
                return self.orig_df[col_name].sample(n=n_examples)

            df = None
            if test_id in ['BINARY_SAME', 'BINARY_OPPOSITE', 'BINARY_IMPLIES', 'BINARY_AND', 'BINARY_OR',
                           'BINARY_XOR', 'BINARY_NUM_SAME', 'BINARY_TWO_OTHERS_MATCH']:
                if len(cols) == 2 and cols[0] in self.binary_cols and cols[1] in self.binary_cols:
                    v0_0, v0_1 = self.orig_df[cols[0]].dropna().unique()
                    v1_0, v1_1 = self.orig_df[cols[1]].dropna().unique()
                    df_v00 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_0) &
                        (self.orig_df[cols[1]] == v1_0)].head(n_examples // 4)
                    df_v01 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_0) &
                        (self.orig_df[cols[1]] == v1_1)].head(n_examples // 4)
                    df_v10 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_1) &
                        (self.orig_df[cols[1]] == v1_0)].head(n_examples // 4)
                    df_v11 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_1) &
                        (self.orig_df[cols[1]] == v1_1)].head(n_examples // 4)
                    df = pd.concat([df_v00, df_v01, df_v10, df_v11])
                elif len(cols) == 3 and cols[0] in self.binary_cols and cols[1] in self.binary_cols and cols[2] in self.binary_cols:
                    v0_0, v0_1 = self.orig_df[cols[0]].dropna().unique()
                    v1_0, v1_1 = self.orig_df[cols[1]].dropna().unique()
                    v2_0, v2_1 = self.orig_df[cols[2]].dropna().unique()
                    df_v000 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_0) &
                        (self.orig_df[cols[1]] == v1_0) &
                        (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                    df_v001 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_0) &
                        (self.orig_df[cols[1]] == v1_0) &
                        (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                    df_v010 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_0) &
                        (self.orig_df[cols[1]] == v1_1) &
                        (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                    df_v011 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_0) &
                        (self.orig_df[cols[1]] == v1_1) &
                        (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                    df_v100 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_1) &
                        (self.orig_df[cols[1]] == v1_0) &
                        (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                    df_v101 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_1) &
                        (self.orig_df[cols[1]] == v1_0) &
                        (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                    df_v110 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_1) &
                        (self.orig_df[cols[1]] == v1_1) &
                        (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                    df_v111 = self.orig_df[
                        (self.orig_df[cols[0]] == v0_1) &
                        (self.orig_df[cols[1]] == v1_1) &
                        (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                    df = pd.concat([df_v000, df_v001, df_v010, df_v011, df_v100, df_v101, df_v110, df_v111])
                elif len(cols) == 3 and cols[2] in self.binary_cols:
                    v0, v1 = self.orig_df[cols[2]].dropna().unique()
                    df_v0 = self.orig_df[
                        (self.orig_df[cols[2]] == v0)].head(n_examples // 2)
                    df_v1 = self.orig_df[
                        (self.orig_df[cols[2]] == v1)].head(n_examples // 2)
                    df = pd.concat([df_v0, df_v1])
            elif test_id in ['MATCHED_MISSING']:
                col_name_1, col_name_2 = cols[0], cols[1]
                df_v_null = self.orig_df[cols][
                    (self.orig_df[col_name_1].isna())].head(n_examples // 2)
                df_v_non_null = self.orig_df[cols][
                    (self.orig_df[col_name_1].notna())].head(n_examples // 2)
                df = pd.concat([df_v_null, df_v_non_null])
            elif test_id in ['MATCHED_ZERO', 'MATCHED_ZERO_MISSING', 'ALL_ZERO_OR_ALL_NON_ZERO']:
                col_name_1 = cols[0]
                df_v_zero = self.orig_df[cols][
                    (self.orig_df[col_name_1] == 0)].head(n_examples // 2)
                df_v_non_zero = self.orig_df[cols][
                    (self.orig_df[col_name_1] != 0)].head(n_examples // 2)
                df = pd.concat([df_v_zero, df_v_non_zero])
            elif test_id in ['SAME_OR_CONSTANT']:
                col_name_1, col_name_2 = cols
                df_same = self.orig_df[cols][
                    (self.orig_df[col_name_1] == self.orig_df[col_name_2])].head(n_examples // 2)
                df_not_same = self.orig_df[cols][
                    (self.orig_df[col_name_1] != self.orig_df[col_name_2])].head(n_examples // 2)
                df = pd.concat([df_same, df_not_same])
            elif test_id in ['NEGATIVE']:
                col_name_1 = cols[0]
                df_v_zero = self.orig_df[cols][
                    (self.orig_df[col_name_1] == 0)].head(n_examples // 2)
                df_v_neg = self.orig_df[cols][
                    (self.orig_df[col_name_1] < 0)].head(n_examples // 2)
                df = pd.concat([df_v_zero, df_v_neg])
            elif test_id in ['POSITIVE']:
                col_name_1 = cols[0]
                df_v_zero = self.orig_df[cols][
                    (self.orig_df[col_name_1] == 0)].head(n_examples // 2)
                df_v_pos = self.orig_df[cols][
                    (self.orig_df[col_name_1] > 0)].head(n_examples // 2)
                df = pd.concat([df_v_zero, df_v_pos])
            elif test_id in ['ALL_POS_OR_ALL_NEG']:
                col_name_1 = cols[0]
                df_v_pos = self.orig_df[cols][
                    (self.orig_df[col_name_1] > 0)].head(n_examples // 2)
                df_v_neg = self.orig_df[cols][
                    (self.orig_df[col_name_1] < 0)].head(n_examples // 2)
                df = pd.concat([df_v_pos, df_v_neg])
            if df is None:
                # Test test_results_df does not have duplicate values in the index
                assert len(self.test_results_df.index) == len(set(self.test_results_df.index))
                df = self.orig_df[cols]
                for col in cols:
                    sub_df = self.orig_df.loc[df.index]
                    mask = sub_df[col].notna()
                    if mask.tolist().count(True) >= 5:
                        df = df[mask]
            if len(df) == 0:  # This shouldn't happen; the column should only be added if there are some patterns
                return None

            if not is_patterns:
                sub_df = self.test_results_df.loc[df.index]
                mask = sub_df[results_col_name] == 0
                df = df[mask]

            df.index = [x[0] if type(x) == tuple else x for x in df.index]
            row_idxs = df.index
            if not is_patterns:
                df = self.orig_df.loc[row_idxs][list(self.col_to_original_cols_dict[results_col_name])]
            if group_results:
                start_point = np.random.randint(0, len(df)-n_examples)
                return df.iloc[start_point: start_point + n_examples]
            else:
                return df.sample(n=min(len(df), n_examples))
        else:
            good_indexes = [1] * self.num_rows
            for test_id in self.get_test_list():
                results_col_name = self.get_results_col_name(test_id, col_name)
                if results_col_name in self.test_results_df.columns:
                    good_indexes = good_indexes & (self.test_results_df[results_col_name] == 0)
            return self.orig_df.loc[good_indexes][col_name].sample(n=n_examples)

    def get_results_by_row_id(self, row_num):
        """
        Returns a list of tuples, with each tuple containing a test ID, and column name, for all issues flagged in the
        specified row.
        """

        if self.test_results_df is None:
            return []

        if row_num > len(self.test_results_df):
            print(f"Cannot display results for row {row_num}. {len(self.test_results_df)} rows available.")
            return []

        row = self.test_results_df.iloc[row_num]
        issues_list = []
        for c in row.index:
            if " -- " not in c:
                continue
            if row[c] == 0:
                continue
            test_id, col_name = c.split(" -- ")
            test_id = test_id.replace("TEST ", "")
            col_name = col_name.replace(" RESULT", "")
            issues_list.append((test_id, col_name))
        return issues_list

    def plot_final_scores_distribution_by_row(self):
        """
        Display a probability plot and histogram representing the distribution of final scores by row.
        """
        final_scores_df = self.test_results_df.copy()
        final_scores_df = final_scores_df.sort_values('FINAL SCORE', ascending=True)
        final_scores_df['Rank'] = range(self.num_rows)
        plt.subplots(figsize=(5, 3))
        s = sns.scatterplot(data=final_scores_df, x='Rank', y='FINAL SCORE')
        s.set_title("Distribution of Scores by Row, ordered lowest to highest scores")
        plt.show()

        plt.subplots(figsize=(5, 3))
        s = sns.histplot(data=final_scores_df, x='FINAL SCORE')
        s.set_title("Distribution of Scores per Row")
        plt.show()

        plt.subplots(figsize=(5, 3))
        s = sns.histplot(data=final_scores_df[final_scores_df['FINAL SCORE'] > 0], x='FINAL SCORE')
        s.set_title("Distribution of Scores per Row (Excluding Scores of 0)")
        plt.show()

    def plot_final_scores_distribution_by_feature(self):
        """
        Display a bar plot representing the distribution of final scores by feature.
        """
        final_scores_series = pd.Series(self.test_results_by_column_np.sum(axis=0)).sort_values(ascending=True).values
        final_scores_df = pd.DataFrame({'Feature': self.orig_df.columns,
                                        'Total Scores': final_scores_series})
        plt.subplots(figsize=(5, len(final_scores_df)*0.25))
        s = sns.barplot(data=final_scores_df, orient='h', y='Feature', x='Total Scores')
        s.set_title("Distribution of Total Scores per Column")
        plt.show()

    def plot_final_scores_distribution_by_test(self):
        """
        Display a bar plot representing the distribution of final scores by test.
        """
        scores_by_test = pd.DataFrame(self.results_summary_df.groupby('Test ID')['Number of Exceptions'].sum().sort_values())
        scores_by_test['Test ID'] = scores_by_test.index

        plt.subplots(figsize=(5, len(scores_by_test) * 0.25))
        s = sns.barplot(data=scores_by_test, orient='h', y='Test ID', x='Number of Exceptions')
        s.set_title("Distribution of Scores per Test")
        plt.show()

    def __plot_distribution(self, test_id, col_name, show_exceptions):
        plt.subplots(figsize=(5, 3))
        s = sns.histplot(data=self.orig_df, x=col_name, color='blue', bins=100)
        if show_exceptions:
            s.set_title(f"Distribution of {col_name} (Flagged values in red)")
        else:
            s.set_title(f"Distribution of {col_name}")

        # Find the flagged values and identify them on the plot
        if show_exceptions:
            results_col_name = self.get_results_col_name(test_id, col_name)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            flagged_vals = self.orig_df.loc[flagged_idxs, col_name].values
            for v in flagged_vals:
                s.axvline(v, color='red')
        plt.show()

    def __plot_scatter_plot(self, test_id, cols, columns_set, show_exceptions):
        col_name_1, col_name_2 = cols
        fig, ax = plt.subplots(figsize=(4, 4))
        s = sns.scatterplot(data=self.orig_df, x=col_name_1, y=col_name_2, color='blue', alpha=0.2, label='Not Flagged')
        if show_exceptions:
            s.set_title(f'Distribution of "{col_name_1}" and "{col_name_2}" (Flagged values in red)')
        else:
            s.set_title(f'Distribution of "{col_name_1}" and "{col_name_2}"')

        # Find the flagged values and identify them on the plot
        if show_exceptions:
            results_col_name = self.get_results_col_name(test_id, columns_set)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            s = sns.scatterplot(data=self.orig_df.loc[flagged_idxs], x=col_name_1, y=col_name_2, color='red', label='Flagged')

        ax.tick_params(axis='x', labelrotation=45)
        if show_exceptions:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            plt.legend().remove()
        plt.show()

    def __plot_count_plot(self, column_name):
        plt.subplots(figsize=(4, 4))
        s = sns.countplot(orient='h', y=self.orig_df[column_name].fillna("NONE"))
        s.set_title(f"Counts of unique values in {column_name}")
        plt.show()

    def __plot_heatmap(self, test_id, cols):
        col_name_1, col_name_2 = cols
        plt.subplots(figsize=(4, 4))

        # todo: we may wish to include null as well, but need special handling below
        vals1 = self.orig_df[col_name_1].dropna().unique()
        vals2 = self.orig_df[col_name_2].dropna().unique()
        counts_arr = []
        for v1 in vals1:
            row_arr = []
            for v2 in vals2:
                row_arr.append(len(self.orig_df[(self.orig_df[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]))
            counts_arr.append(row_arr)
        df = pd.DataFrame(counts_arr, index=vals1, columns=vals2)
        s = sns.heatmap(df, cmap='Blues', linewidths=1.1, linecolor='black', annot=True, fmt="d")
        s.set_title(f"Counts of unique values in {col_name_1} and {col_name_2}")
        plt.show()

    def __draw_row_rank_plot(self, test_id, col_name, show_exceptions):
        fig, ax = plt.subplots(figsize=(4, 4))
        s = sns.scatterplot(x=self.orig_df.index, y=self.orig_df[col_name], color='blue', alpha=0.2, label='Not Flagged')
        if show_exceptions:
            s.set_title(f'Distribution of "{col_name}" (Flagged values in red)')
        else:
            s.set_title(f'Distribution of "{col_name}"')

        # Find the flagged values and identify them on the plot
        if show_exceptions:
            results_col_name = self.get_results_col_name(test_id, col_name)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            flagged_vals = self.orig_df.loc[flagged_idxs][col_name].values
            s = sns.scatterplot(x=flagged_idxs[0], y=flagged_vals, color='red', label="Flagged")
        s.set_xlabel("Row Number")
        ax.tick_params(axis='x', labelrotation=45)
        plt.legend().remove()
        plt.show()

    def __draw_box_plots(self, test_id, cols, columns_set):
        # The first column is the string/binary value and the second is the numeric/date value
        # todo: this does not colour the outliers. We may wish to draw a histogram next to it, but this is kludgy.
        # results_col_name = self.get_results_col_name(test_id, columns_set)
        # col_names = self.col_to_original_cols_dict[results_col_name]
        s = sns.boxplot(data=self.orig_df, orient='h', y=cols[0], x=cols[1])
        plt.show()

        # Also draw a histogram of the relevant classes.
        results_col_name = self.get_results_col_name(test_id, columns_set)
        results_col = self.test_results_df[results_col_name]
        flagged_idxs = np.where(results_col)
        flagged_df = self.orig_df.loc[flagged_idxs]
        vals = flagged_df[cols[0]].unique()
        nvals = len(vals)
        fig, ax = plt.subplots(nrows=1, ncols=nvals, figsize=(nvals*4, 4))
        for v_idx, v in enumerate(vals):
            sub_df = self.orig_df[self.orig_df[cols[0]] == v]
            sub_flagged_df = flagged_df[flagged_df[cols[0]] == v]
            if nvals == 1:
                curr_ax = ax
            else:
                curr_ax = ax[v_idx]
            s = sns.histplot(data=sub_df, x=cols[1], color='blue', bins=100, ax=curr_ax)
            flagged_vals = sub_flagged_df[cols[1]].values
            for fv in flagged_vals:
                s.axvline(fv, color='red')
            s.set_title(f"Distribution of {cols[1]} where {cols[0]} is {v} (Flagged values in red)")
        plt.show()

    def __draw_sample_dataframe(self, df, test_id, cols, display_info, is_patterns):
        if len(cols) == 1:
            col_name = cols[0]
        if len(cols) == 2:
            col_name_1, col_name_2 = cols
        if len(cols) == 3:
            col_name_1, col_name_2, col_name_3 = cols
        if len(cols) > 3:
            col_set = cols
            source_cols = col_set[:-1]

        # Add the additional columns to explain the relationship
        df = df.copy()
        if test_id in ['LARGER_THAN_SUM', 'CONSTANT_SUM']:
            df['SUM'] = df[col_name_1] + df[col_name_2]
        elif test_id in ['SUM_OF_COLUMNS']:
            if display_info and display_info['operation'] == "plus":
                df[f"SUM PLUS {display_info['amount']}"] = df[source_cols].sum(axis=1) + display_info['amount']
            elif display_info and display_info['operation'] == "times":
                df[f"SUM TIMES {display_info['amount']}"] = df[source_cols].sum(axis=1) * display_info['amount']
            else:
                df['SUM'] = df[source_cols].sum(axis=1)
        elif test_id in ['SIMILAR_TO_DIFF', 'LARGER_THAN_ABS_DIFF', 'CONSTANT_DIFF']:
            df['ABSOLUTE DIFFERENCE'] = abs(df[col_name_1] - df[col_name_2])
        elif test_id in ['SIMILAR_TO_PRODUCT', 'CONSTANT_PRODUCT']:
            df['PRODUCT'] = df[col_name_1] * df[col_name_2]
        elif test_id in ['SIMILAR_TO_RATIO', 'CONSTANT_RATIO', 'EVEN_MULTIPLE']:
            df['DIVISION RESULTS'] = df[col_name_1] / df[col_name_2]
        elif test_id in ['MEAN_OF_COLUMNS']:
            df['MEAN'] = df[source_cols].mean(axis=1)
        elif test_id in ['MIN_OF_COLUMNS']:
            df['MIN'] = df[source_cols].min(axis=1)
        elif test_id in ['MAX_OF_COLUMNS']:
            df['MAX'] = df[source_cols].max(axis=1)
        elif test_id in ['RARE_PAIRS_FIRST_WORD_VAL', 'LARGE_GIVEN_PREFIX', 'SMALL_GIVEN_PREFIX']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            df[f'{col_name_1} FIRST WORD'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['LEADING_WHITESPACE']:
            df['NUM LEADING SPACES'] = df[col_name].astype(str).str.len() - df[col_name].astype(str).str.lstrip(' ').str.len()
        elif test_id in ['TRAILING_WHITESPACE']:
            df['NUM TRAILING SPACES'] = df[col_name].astype(str).str.len() - df[col_name].astype(str).str.rstrip(' ').str.len()
        elif test_id in ['MULTIPLE_OF_CONSTANT']:
            if is_patterns:
                df['NUM MULTIPLES'] = round(df[col_name] / display_info['value'])
            else:
                df['NUM MULTIPLES'] = df[col_name] / display_info['value']
        elif test_id in ['UNUSUAL_DAY_OF_WEEK']:
            df['Day of Week'] = df[col_name].dt.dayofweek
        elif test_id in ['UNUSUAL_DAY_OF_MONTH']:
            df['Day of Month'] = df[col_name].dt.day
        elif test_id in ['UNUSUAL_MONTH']:
            df['Month'] = df[col_name].dt.month
        elif test_id in ['UNUSUAL_HOUR_OF_DAY']:
            df['Hour'] = df[col_name].dt.hour
        elif test_id in ['UNUSUAL_MINUTES']:
            df['Minutes'] = df[col_name].dt.minute
        elif test_id in ['CONSTANT_GAP', 'LARGE_GAP', 'SMALL_GAP', 'LATER']:
            df['Gap'] = df[col_name_2] - df[col_name_1]
        elif test_id in ['NUMBER_ALPHA_CHARS']:
            df['Num Alpha Chars'] = df[col_name].astype(str).apply(lambda x: len([e for e in x if e.isalpha()]))
        elif test_id in ['NUMBER_NUMERIC_CHARS']:
            df['Num Numeric Chars'] = df[col_name].astype(str).apply(lambda x: len([e for e in x if e.isdigit()]))
        elif test_id in ['NUMBER_ALPHANUMERIC_CHARS']:
            df['Num Alpha-Numeric Chars'] = df[col_name].astype(str).apply(lambda x: len([e for e in x if e.isalnum()]))
        elif test_id in ['NUMBER_NON-ALPHANUMERIC_CHARS']:
            df['Num Alpha-Numeric Chars'] = df[col_name].astype(str).apply(lambda x: len([e for e in x if not e.isalnum()]))
        elif test_id in ['NUMBER_CHARS', 'MANY_CHARS', 'FEW_CHARS']:
            df['Num Chars'] = df[col_name].astype(str).str.len()
        elif test_id in ['FIRST_CHAR_ALPHA', 'FIRST_CHAR_NUMERIC', 'FIRST_CHAR_SMALL_SET', 'FIRST_CHAR_UPPERCASE',
                         'FIRST_CHAR_LOWERCASE']:
            df['First Char'] = df[col_name].astype(str).str.lstrip().str.slice(0,1)
        elif test_id in ['LAST_CHAR_SMALL_SET']:
            df['Last Char'] = df[col_name].astype(str).str.rstrip().str[-1:]
        elif test_id in ['FIRST_WORD_SMALL_SET']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            df['First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['LAST_WORD_SMALL_SET']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            df['Last Word'] = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['NUMBER_WORDS']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            df['Num Words'] = [len(x) for x in word_arr]
        elif test_id in ['LONGEST_WORDS']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            word_lens_arr = [[len(w) for w in x] for x in word_arr]
            df['Longest Word Len'] = [max(x) if len(x) > 0 else 0 for x in word_lens_arr]
        elif test_id in ['RARE_PAIRS_FIRST_CHAR', 'SAME_FIRST_CHARS']:
            df[f'{col_name_1} First Char'] = df[col_name_1].astype(str).str[:1]
            df[f'{col_name_2} First Char'] = df[col_name_2].astype(str).str[:1]
        elif test_id in ['RARE_PAIRS_FIRST_WORD', 'SAME_FIRST_WORD']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            df[f'{col_name_1} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            col_vals = df[col_name_2].astype(str).apply(replace_special_with_space)
            df[f'{col_name_2} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['SAME_LAST_WORD']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            df[f'{col_name_1} Last Word'] = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
            col_vals = df[col_name_2].astype(str).apply(replace_special_with_space)
            df[f'{col_name_2} Last Word'] = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['SIMILAR_NUM_CHARS']:
            df[f'{col_name_1} Num Chars'] = df[col_name_1].astype(str).str.len()
            df[f'{col_name_2} Num Chars'] = df[col_name_2].astype(str).str.len()
        elif test_id in ['SIMILAR_NUM_WORDS']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            df[f'{col_name_1} Num Words'] = [len(x) for x in word_arr]
            col_vals = df[col_name_2].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            df[f'{col_name_2} Num Words'] = [len(x) for x in word_arr]
        elif test_id in ['SMALL_VS_CORR_COLS', 'LARGE_VS_CORR_COLS']:
            for c in display_info['cluster']:
                df[c] = self.orig_df[c].loc[df.index]
            df = df[list(df.columns[1:]) + [df.columns[0]]]  # Ensure the relevant column is the rightmost
        elif test_id in ['MISSING_VALUES_PER_ROW']:
            df['Number Missing'] = df.isna().sum(axis=1)
        elif test_id in ['ZERO_VALUES_PER_ROW']:
            df['Number Zero'] = df.applymap(lambda x: (x is None) or (x == 0)).sum(axis=1)
        elif test_id in ['UNIQUE_VALUES_PER_ROW']:
            df['Number Uniuqe'] = df.apply(lambda x: len(set(x)), axis=1)
        elif test_id in ['NEGATIVE_VALUES_PER_ROW']:
            df['Number Negative'] = df.applymap(lambda x: isinstance(x, numbers.Number) and x < 0).sum(axis=1)

        df = df.sort_index()
        if is_notebook():
            display(df)
        else:
            print(df)

    def __draw_sample_set_rows(self, df):
        row_id = df.index[0]
        row_start = max(0, row_id-5)
        row_end = min(self.num_rows, row_id+5)
        neighborhood_df = self.orig_df.iloc[row_start:row_end][df.columns]
        if is_notebook():
            display(neighborhood_df)
        else:
            print(neighborhood_df)

    def display_examples(self, test_id, cols, columns_set, is_patterns, display_info):

        # Do not show examples for some tests
        if is_patterns and test_id in ['UNIQUE_VALUES']:
            print("Examples not shown for this pattern.")
            return

        if test_id in ['MISSING_VALUES_PER_ROW', 'ZERO_VALUES_PER_ROW', 'UNIQUE_VALUES_PER_ROW',
                       'NEGATIVE_VALUES_PER_ROW']:
            print("Examples not shown for this pattern.")
            return

        show_consecutive = test_id in ['PREV_VALUES_DT', 'COLUMN_ORDERED_ASC', 'COLUMN_ORDERED_DESC',
                                       'COLUMN_TENDS_ASC','COLUMN_TENDS_DESC', 'SIMILAR_PREVIOUS', 'RUNNING_SUM',
                                       'GROUPED_STRINGS', 'GROUPED_STRINGS_BY_NUMERIC']
        consecutive_str = ""
        if show_consecutive:
            consecutive_str = " (showing a consecutive set of rows)"

        print()
        if is_patterns:
            print(f"Examples{consecutive_str}:")
        else:
            print(f"Examples of values not flagged{consecutive_str}:")

        if show_consecutive:
            vals = self.get_sample_not_flagged(test_id, columns_set, group_results=True, is_patterns=is_patterns)
        else:
            vals = self.get_sample_not_flagged(test_id, columns_set, group_results=False, is_patterns=is_patterns)
        self.__draw_sample_dataframe(vals, test_id, cols, display_info, is_patterns)

    def draw_results_plots(self, test_id, cols, columns_set, show_exceptions, display_info):
        if test_id in ['UNUSUAL_NUMERIC', 'FEW_NEIGHBORS', 'FEW_WITHIN_RANGE', 'VERY_SMALL', 'VERY_LARGE',
                       'VERY_SMALL_ABS', 'LESS_THAN_ONE', 'GREATER_THAN_ONE', 'NON_ZERO', 'POSITIVE', 'NEGATIVE',
                       'EARLY_DATES', 'LATE_DATES']:
            self.__plot_distribution(test_id, cols[0], show_exceptions)

        if test_id in ['LARGER', 'MUCH_LARGER', 'SIMILAR_WRT_RATIO', 'SIMILAR_WRT_DIFF', 'SIMILAR_TO_INVERSE',
                       'SIMILAR_TO_NEGATIVE', 'CONSTANT_SUM', 'CONSTANT_DIFF', 'CONSTANT_PRODUCT', 'CONSTANT_RATIO',
                       'CORRELATED_FEATURES', 'RARE_COMBINATION', 'SAME_VALUES', 'LARGE_GIVEN_DATE', 'SMALL_GIVEN_DATE']:
            # todo: also provide the spearman correlation for these cases
            self.__plot_scatter_plot(test_id, cols, columns_set, show_exceptions)

        if test_id in ['LARGER_THAN_SUM', 'SIMILAR_TO_DIFF', 'LARGER_THAN_ABS_DIFF', 'SIMILAR_TO_PRODUCT',
                       'SIMILAR_TO_RATIO']:
            df = self.orig_df.copy()
            col_name_1, col_name_2, col_name_3 = cols
            # todo: these are duplicated in __draw_sample_dataframe()
            if test_id in ['LARGER_THAN_SUM']:
                calculated_col = 'SUM'
                df[calculated_col] = df[col_name_1] + df[col_name_2]
            elif test_id in ['SIMILAR_TO_DIFF', 'LARGER_THAN_ABS_DIFF']:
                calculated_col = 'Absolute Difference'
                df[calculated_col] = abs(df[col_name_1] - df[col_name_2])
            elif test_id in ['SIMILAR_TO_PRODUCT']:
                calculated_col = 'PRODUCT'
                df[calculated_col] = df[col_name_1] * df[col_name_2]
            elif test_id in ['SIMILAR_TO_RATIO']:
                calculated_col = 'Division Results'
                df[calculated_col] = df[col_name_1] / df[col_name_2]
            plt.subplots(figsize=(4, 4))
            s = sns.scatterplot(
                data=df,
                x=col_name_3,
                y=calculated_col,
                color='blue',
                alpha=0.2,
                label='Not Flagged')
            if show_exceptions:
                s.set_title(f'Distribution of "{col_name_3}" and "{calculated_col}" (Flagged values in red)')
            else:
                s.set_title(f'Distribution of "{col_name_3}" and "{calculated_col}"')

            # Find the flagged values and identify them on the plot
            if show_exceptions:
                result_col_name = self.get_results_col_name(test_id, columns_set)
                results_col = self.test_results_df[result_col_name]
                flagged_idxs = np.where(results_col)
                sns.scatterplot(
                    data=df.loc[flagged_idxs],
                    x=col_name_3,
                    y=calculated_col,
                    color='red',
                    label='Flagged')
            plt.legend().remove()
            plt.show()

        if test_id in ['RARE_VALUES']:
            self.__plot_count_plot(cols[0])

        if test_id in ['RARE_PAIRS', 'BINARY_SAME', 'BINARY_OPPOSITE', 'BINARY_IMPLIES']:
            self.__plot_heatmap(test_id, cols)

        if test_id in ['COLUMN_ORDERED_ASC', 'COLUMN_ORDERED_DESC', 'COLUMN_TENDS_ASC', 'COLUMN_TENDS_DESC',
                       'SIMILAR_PREVIOUS']:
            self.__draw_row_rank_plot(test_id, cols[0], show_exceptions)

        if test_id in ['SMALL_GIVEN_VALUE', 'LARGE_GIVEN_VALUE', 'BINARY_MATCHES_VALUE']:
            self.__draw_box_plots(test_id, cols, columns_set)

        if test_id in ['UNUSUAL_DAY_OF_WEEK']:
            sns.countplot(x=self.orig_df[cols[0]].dt.dayofweek)
            plt.show()

        if test_id in ['UNUSUAL_DAY_OF_MONTH']:
            sns.countplot(x=self.orig_df[cols[0]].dt.day)
            plt.show()

        if test_id in ['UNUSUAL_MONTH']:
            sns.countplot(x=self.orig_df[cols[0]].dt.month)
            plt.show()

        if test_id in ['UNUSUAL_HOUR_OF_DAY']:
            sns.countplot(x=self.orig_df[cols[0]].dt.hour)
            plt.show()

        if test_id in ['UNUSUAL_MINUTES']:
            sns.countplot(x=self.orig_df[cols[0]].dt.minute)
            plt.show()

        elif test_id in ['CONSTANT_GAP', 'LARGE_GAP', 'SMALL_GAP', 'LATER']:
            sns.countplot(x=((self.orig_df[cols[1]] - self.orig_df[cols[0]]).dt.days))
            plt.show()

        elif test_id in ['RARE_PAIRS_FIRST_CHAR']:
            df2 = self.orig_df[cols].copy()
            df2[f'{cols[0]} First Char'] = df2[cols[0]].astype(str).str[:1]
            df2[f'{cols[1]} First Char'] = df2[cols[1]].astype(str).str[:1]
            counts_data = pd.crosstab(df2[f'{cols[0]} First Char'], df2[f'{cols[1]} First Char'])
            s = sns.heatmap(counts_data, cmap="Blues", annot=True, fmt='g')
            s.set_title(f"Counts by First Characters of {cols[0]} and {cols[1]}")
            plt.show()

        elif test_id in ['RARE_PAIRS_FIRST_WORD']:
            df2 = self.orig_df[cols].copy()
            col_vals = df2[cols[0]].astype(str).apply(replace_special_with_space)
            df2[f'{cols[0]} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            col_vals = df2[cols[1]].astype(str).apply(replace_special_with_space)
            df2[f'{cols[1]} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            counts_data = pd.crosstab(df2[f'{cols[0]} First Word'], df2[f'{cols[1]} First Word'])
            s = sns.heatmap(counts_data, cmap="Blues", annot=True, fmt='g')
            s.set_title(f"Counts by First Words of {cols[0]} and {cols[1]}")
            plt.show()

        elif test_id in ['CORRELATED_ALPHA_ORDER']:
            df2 = self.orig_df[cols].copy()
            df2[cols[0]] = self.orig_df[cols[0]].rank(pct=True)
            df2[cols[1]] = self.orig_df[cols[1]].rank(pct=True)
            s = sns.scatterplot(data=df2, x=cols[0], y=cols[1])
            s.set_title("Values by Alphabetic Order")

            # Find the flagged values and identify them on the plot
            if show_exceptions:
                results_col_name = self.get_results_col_name(test_id, columns_set)
                results_col = self.test_results_df[results_col_name]
                flagged_idxs = np.where(results_col)
                s = sns.scatterplot(data=df2.loc[flagged_idxs], x=cols[0], y=cols[1], color='red', label='Flagged')
            plt.show()

        elif test_id in ['LARGE_GIVEN_PREFIX', 'SMALL_GIVEN_PREFIX']:
            df2 = self.orig_df[cols].copy()
            col_vals = df2[cols[0]].astype(str).apply(replace_special_with_space)
            df2[cols[0]] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            s = sns.boxplot(data=df2, orient='h', y=cols[0], x=cols[1])

            # Also draw a histogram of the relevant classes.
            results_col_name = self.get_results_col_name(test_id, columns_set)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            flagged_df = self.orig_df.loc[flagged_idxs]
            col_vals = flagged_df[cols[0]].astype(str).apply(replace_special_with_space)
            flagged_df[cols[0]] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            vals = pd.Series(flagged_df[cols[0]].astype(str).apply(replace_special_with_space))
            vals = pd.Series([x[0] if len(x) > 0 else "" for x in vals.str.split()]).unique()
            nvals = len(vals)
            fig, ax = plt.subplots(nrows=1, ncols=nvals, figsize=(nvals*4, 4))
            for v_idx, v in enumerate(vals):
                sub_df = df2[df2[cols[0]] == v]
                sub_flagged_df = flagged_df[flagged_df[cols[0]] == v]
                if nvals == 1:
                    curr_ax = ax
                else:
                    curr_ax = ax[v_idx]
                s = sns.histplot(data=sub_df, x=cols[1], color='blue', bins=100, ax=curr_ax)
                flagged_vals = sub_flagged_df[cols[1]].values
                for fv in flagged_vals:
                    s.axvline(fv, color='red')
                s.set_title(f"Distribution of {cols[1]} where the first word of {cols[0]} is {v} (Flagged values in red)")
                plt.show()

        elif test_id in ['SMALL_AVG_RANK_PER_ROW', 'LARGE_AVG_RANK_PER_ROW']:
            t_df = pd.DataFrame({"Avg. Percentiles": display_info['percentiles']})
            fig, ax = plt.subplots(figsize=(6, 2))
            s = sns.histplot(data=t_df, x='Avg. Percentiles')
            for fv in display_info['flagged_vals']:
                ax.axvline(fv, color='r')
            s.set_title("Mean Percentile of Numeric Values by Row")
            plt.show()

    def display_detailed_results(
            self,
            test_id_list=None,
            col_name_list=None,
            show_patterns=True,
            short_list=True,
            plot_results=True):
        """
        Go through each test, and each feature, and present a detailed description of each.
        """
        # todo: for each column, give the number of missing values
        # todo: where possible, show a plot. Can do for very large, very small, few neighbors, unusual combinations, etc

        def print_test_header(test_id):
            nonlocal printed_test_header
            nonlocal test_id_list

            if printed_test_header:
                return
            if len(test_id_list) == 1 or (self.execute_list is not None and len(self.execute_list) == 1):
                return
            print()
            print(stars)
            print(test_id)
            print(stars)
            printed_test_header = True

        def print_column_header(col_name):
            print()
            print(hyphens)
            print(f"Column(s): {col_name}")

        if (self.orig_df is None) or (len(self.orig_df) == 0):
            print("Empty dataset")
            return

        if ((self.patterns_df is None) or (len(self.patterns_df) == 0)) and \
            ((self.results_summary_df is None) or (len(self.results_summary_df) == 0)):
            return

        if (test_id_list is None) and (col_name_list is None):
            if (len(self.results_summary_df) > 200):
                print()
                print((f"{len(self.results_summary_df)} issues were identified. This is beyond the limit to display "
                    "all at once. Try specifying tests and/or columns to be displayed here."))
                return
            if (show_patterns and ((len(self.results_summary_df) + len(self.patterns_df)) > 200)):
                print()
                print((f"{len(self.results_summary_df) + len(self.patterns_df)} patterns and issues were identified. " 
                        "This is beyond the limit to display all at once. Try specifying tests and/or columns to be "
                       "displayed here."))
                return

        if test_id_list:
            print()
            print(f"Displaying results for tests: {test_id_list}")
        if col_name_list:
            print()
            print(f"Displaying results for columns: {col_name_list}")

        if test_id_list is None:
            test_id_list = self.get_test_list()
        if col_name_list is None:
            col_name_list = self.orig_df.columns

        stars = "******************************************************************************"
        hyphens = '----------------------------------------------------------------------------'
        for test_id in test_id_list:
            if self.execute_list and test_id not in self.execute_list:
                continue
            if self.exclude_list and test_id in self.exclude_list:
                continue

            printed_test_header = False

            sub_patterns_test = self.patterns_df[self.patterns_df['Test ID'] == test_id]
            sub_results_summary_test = self.results_summary_df[self.results_summary_df['Test ID'] == test_id]

            # Display patterns that have no exception
            if show_patterns and not (short_list and test_id not in self.get_patterns_shortlist()):
                for columns_set in sub_patterns_test['Column(s)'].values:
                    sub_patterns = self.patterns_df[(self.patterns_df['Test ID'] == test_id) &
                                                    (self.patterns_df['Column(s)'] == columns_set)]
                    if len(set(col_name_list).intersection(set(columns_set))) == 0:
                        continue
                    if len(sub_patterns) > 0:
                        print_test_header(test_id)
                        print_column_header(columns_set)
                        print("Pattern found (without exceptions)")
                        print(sub_patterns.iloc[0]['Description of Pattern'])
                        cols = [x.lstrip('"').rstrip('"') for x in columns_set.split(" AND ")]
                        self.display_examples(
                            test_id,
                            cols,
                            columns_set,
                            is_patterns=True,
                            display_info=sub_patterns.iloc[0]['Display Information'])
                        if plot_results:
                            self.draw_results_plots(
                                test_id,
                                cols,
                                columns_set,
                                show_exceptions=False,
                                display_info=sub_patterns.iloc[0]['Display Information'])

            # Display patterns with exceptions
            for columns_set in sub_results_summary_test['Column(s)'].values:
                # If columns_set_arr is specified, only report issues with some overlap of columns with columns_set_arr.
                # The columns_set in the issues dataframe may be a single string. If so, convert to array.
                issue_columns_arr = self.col_to_original_cols_dict[self.get_results_col_name(test_id, columns_set)]
                if len(set(col_name_list).intersection(set(issue_columns_arr))) == 0:
                    continue

                sub_summary = self.results_summary_df[(self.results_summary_df['Test ID'] == test_id) &
                                                      (self.results_summary_df['Column(s)'] == columns_set)]
                if len(sub_summary) == 0:
                    continue

                print_test_header(test_id)
                print_column_header(columns_set)
                print("Issue index: ", sub_summary.index[0])
                print("A strong pattern, and exceptions to the pattern, were found.")
                if test_id in ['PREV_VALUES_DT', 'DECISION_TREE_REGRESSOR', 'DECISION_TREE_CLASSIFIER']:
                    # These display a decision tree, so the formatting must be preserved.
                    print(sub_summary.iloc[0]['Description of Pattern'])
                else:
                    multiline_desc = wrap(sub_summary.iloc[0]['Description of Pattern'], 100)
                    print('\n'.join(multiline_desc))
                num_exceptions = sub_summary.iloc[0]['Number of Exceptions']
                print((f"Number of exceptions: {num_exceptions} "
                      f"({num_exceptions * 100.0 / self.num_rows:.4f}% of rows)"))

                # Display examples not flagged
                result_col_name = self.get_results_col_name(test_id, columns_set)
                cols = self.col_to_original_cols_dict[result_col_name]
                self.display_examples(
                    test_id,
                    cols,
                    columns_set,
                    is_patterns=False,
                    display_info=sub_summary.iloc[0]['Display Information'])

                print()
                print("Examples of flagged values:")
                display_cols = list(self.col_to_original_cols_dict[self.get_results_col_name(test_id, columns_set)])
                flagged_df = self.get_rows_flagged(test_id, columns_set).head(10)
                if flagged_df is None:
                    continue
                self.__draw_sample_dataframe(
                    flagged_df[display_cols],
                    test_id,
                    cols,
                    display_info=sub_summary.iloc[0]['Display Information'],
                    is_patterns=False)

                # For some tests, we display the rows before and after the flagged rows as well, to provide context
                if test_id in ['PREV_VALUES_DT', 'COLUMN_ORDERED_ASC', 'COLUMN_ORDERED_DESC', 'COLUMN_TENDS_ASC',
                               'COLUMN_TENDS_DESC', 'SIMILAR_PREVIOUS', 'RUNNING_SUM', 'GROUPED_STRINGS',
                               'GROUPED_STRINGS_BY_NUMERIC']:
                    print()
                    print("Showing the first flagged example with the 5 rows before and 5 rows after:")
                    self.__draw_sample_set_rows(flagged_df[display_cols])

                # For some tests, we display one or more plots to make the exceptions more clear
                if plot_results:
                    result_col_name = self.get_results_col_name(test_id, columns_set)
                    cols = self.col_to_original_cols_dict[result_col_name]
                    self.draw_results_plots(
                        test_id,
                        cols,
                        columns_set,
                        show_exceptions=True,
                        display_info=sub_summary.iloc[0]['Display Information'])

    # todo: provide a parameter to name the export
    # todo: improve the font
    # todo: add more layers of expanding divs: buy test and by issue
    # todo: add background colour & border per div.
    def export_html(self, test_id_list=None):

        def print_test_header(test_id, section_name, f):
            nonlocal test_id_list
            assert section_name in ['patterns', 'exceptions']

            f.write("<br>" + "\n")
            func_name = f"func_test_div_{section_name}_{test_id}"
            div_name = f"test_div_{section_name}_{test_id}"
            script_str = """
                <script>
                function func_name() {
                    var x = document.getElementById("div_name");
                    if (x.style.display === "none") {
                        x.style.display = "block";
                    } else {
                        x.style.display = "none";
                    }
                }
                </script>
                """
            script_str = script_str.replace("func_name", func_name)
            script_str = script_str.replace("div_name", div_name)
            f.write(script_str + "\n")

            f.write(f'<p onclick="{func_name}()">{test_id}</p>')
            return div_name

        def print_column_header(col_name, f):
            f.write(f"<br>Column(s): {col_name}" + "\n")

        if test_id_list is None:
            test_id_list = self.get_test_list()

        with open("Data_consistency.html", 'w') as f:
            f.write("<html>" + os.linesep)
            f.write("<head>" + os.linesep)
            f.write("</head>" + os.linesep)
            f.write("<body>" + os.linesep)
            f.write("<h1>Data Consistency Check Results</h1>" + "\n")

            f.write("<a href='#Patterns'>Patterns</a><br/>" + "\n")
            f.write("<a href='#Exceptions'>Exceptions</a><br/>" + "\n")

            f.write("<h2 id='Patterns'>Patterns</h2>" + "\n")
            for test_id in test_id_list:
                sub_patterns_test = self.patterns_df[self.patterns_df['Test ID'] == test_id]

                for columns_set in sub_patterns_test['Column(s)'].values:
                    sub_patterns = self.patterns_df[(self.patterns_df['Test ID'] == test_id) &
                                                    (self.patterns_df['Column(s)'] == columns_set)]
                    if len(sub_patterns) > 0:
                        print_test_header(test_id, "patterns", f)
                        print_column_header(columns_set, f)
                        f.write("<p>Pattern found (without exceptions)</p>")
                        f.write(sub_patterns.iloc[0]['Description of Pattern'])

            f.write("<h2 id='Exceptions'>Exceptions</h2>")
            for test_id in test_id_list:
                sub_results_summary_test = self.results_summary_df[self.results_summary_df['Test ID'] == test_id]
                if len(sub_results_summary_test) == 0:
                    continue
                div_name = print_test_header(test_id, 'exceptions', f)
                f.write(f'<div id="{div_name}">')
                for columns_set in sub_results_summary_test['Column(s)'].values:
                    sub_summary = self.results_summary_df[(self.results_summary_df['Test ID'] == test_id) &
                                                          (self.results_summary_df['Column(s)'] == columns_set)]
                    if len(sub_summary) == 0:
                        continue

                    print_column_header(columns_set, f)
                    f.write(f"<br>Issue index: {sub_summary.index[0]}")
                    f.write("<br>A strong pattern, and exceptions to the pattern, were found.<br>")
                    f.write(sub_summary.iloc[0]['Description of Pattern'])
                    num_exceptions = sub_summary.iloc[0]['Number of Exceptions']
                    f.write((f"<br>Number of exceptions: {num_exceptions} "
                            f"({num_exceptions * 100.0 / self.num_rows:.4f}% of rows)"))
                f.write('</div>')

            f.write("</body>")
            f.write("</html>")

    #########################################################################
    # Methods to find relationships between the data and the numbers of issues found.
    #########################################################################

    def plot_columns_vs_final_scores(self):

        def clear_last_plots():
            if nrows == 1:
                for i in range(num_feats, 4):
                    ax[i].set_visible(False)
            else:
                last_col_used = num_feats % 4
                for i in range(last_col_used, 4):
                    ax[nrows-1][i].set_visible(False)

        if self.results_summary_df is None or len(self.results_summary_df) == 0:
            print("No exceptions to plot.")
            return

        df = self.orig_df.copy()
        df['FINAL SCORE'] = self.test_results_df['FINAL SCORE']

        num_feats = len(self.numeric_cols) + len(self.date_cols)
        if num_feats > 50:
            print((f"There are {num_feats} numeric and date features. Displaying only the 50 with the greatest"
                   "correlation with the final score"))
            num_feats = 50
            feats = self.numeric_cols + self.date_cols
            feats = feats[:50]  # todo: get the 50 with the greatest correlation
        else:
            feats = self.numeric_cols + self.date_cols

        if num_feats > 0:
            nrows = math.ceil(num_feats / 4)
            fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(14, 4 * nrows))
            for feat_idx, col_name in enumerate(feats):
                if nrows == 1:
                    cur_ax = ax[feat_idx]
                else:
                    cur_ax = ax[feat_idx // 4][feat_idx % 4]
                s = sns.scatterplot(data=df, x=df[col_name], y=df['FINAL SCORE'], ax=cur_ax)
                s.set(xlabel=None)
                s.set_title(col_name)
            clear_last_plots()
            plt.suptitle("Relationship of features to Final Score (Numeric and Date features)")
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()

        num_feats = len(self.binary_cols) + len(self.string_cols)
        if num_feats > 50:
            print((f"There are {num_feats} numeric and date features. Displaying only the 50 with the greatest "
                   "correlation with the final score"))
            num_feats = 50
            feats = self.binary_cols + self.string_cols
            feats = feats[:50]  # todo: get the 50 with the greatest correlation
        else:
            feats = self.binary_cols + self.string_cols

        if num_feats > 0:
            nrows = math.ceil(num_feats / 4)
            fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(14, 4 * nrows))
            for feat_idx, col_name in enumerate(feats):
                vc = self.orig_df[col_name].value_counts()
                if nrows == 1:
                    cur_ax = ax[feat_idx]
                else:
                    cur_ax = ax[feat_idx // 4][feat_idx % 4]
                if len(vc) > 10:
                    common_vals = []
                    for v_idx in vc.index:
                        if vc[v_idx] > (self.num_rows / 10):
                            common_vals.append(v_idx)
                    if len(common_vals) == 0:
                        continue
                    map_dict = {x: x for x in common_vals}
                    sub_df = df.copy()
                    sub_df[col_name] = df[col_name].map(map_dict)
                    sub_df[col_name] = sub_df[col_name].fillna("Other")
                    s = sns.boxplot(data=sub_df,  x=col_name, y='FINAL SCORE', ax=cur_ax)
                else:
                    s = sns.boxplot(data=df,  x=col_name, y='FINAL SCORE', ax=cur_ax)
                s.set_title(col_name)

            clear_last_plots()
            plt.suptitle("Relationship of features to Final Score (String and Binary features)")
            plt.tight_layout()
            # See https://stackoverflow.com/questions/8248467/tight-layout-doesnt-take-into-account-figure-suptitle
            plt.subplots_adjust(top=0.90)
            plt.show()

    #########################################################################
    # Internal methods to aid in analysing the data
    #########################################################################

    def get_results_col_name(self, test_id, col_name):
        """
        Generates a column name for columns in self.results_df, which represent the results of running one test on one
        column or set of columns.
        """
        return f"TEST {test_id} -- {col_name} RESULT"

    @staticmethod
    def get_col_set_name(col_names):
        # todo: find all cases where this should be used and switch to this
        """
        Given an array of column names, generate a string representation.
        """
        col_name_str = ""
        for c in col_names:
            col_name_str += f'"{c}" AND '
        col_name_str = col_name_str[:-5]
        return col_name_str

    def __calculate_final_scores(self):
        """
        Calculates the final score for each row in the original data. This treats each test on each column equally and
        calculates their count.
        """
        if self.test_results_df is not None:
            self.test_results_df['FINAL SCORE'] = self.test_results_df.sum(axis=1)

    def __output_stats(self):
        """
        Displays an overview of the tests run and its findings.
        """

        if self.n_tests_executed == 0:
            print("No tests executed.")
            return

        print()
        print("Data consistency check complete.")
        print(f"Analysed {self.num_rows:,} rows, {len(self.orig_df.columns)} columns")
        print(f"Executed {self.n_tests_executed} tests.")
        print()
        print('Patterns without Exceptions:')
        print(f"Found {len(self.patterns_df)} patterns without exceptions")
        print((f"{self.patterns_df['Test ID'].nunique()} tests "
               f"({self.patterns_df['Test ID'].nunique() * 100.0 / self.n_tests_executed:.2f}% of tests) "
               f"identified at least one pattern without exceptions each."))
        print()
        print('Patterns with Exceptions:')
        print(f"Found {len(self.results_summary_df)} patterns with exceptions")
        print((f"{self.results_summary_df['Test ID'].nunique()} tests "
               f"({self.results_summary_df['Test ID'].nunique() * 100.0 / self.n_tests_executed:.2f}% of tests) "
               f"flagged at least one exception each."))
        print((f"Flagged {len(self.test_results_df[self.test_results_df['FINAL SCORE'] > 0]):,} row(s) with at "
               f"least one exception."))
        print(f"Flagged {(self.test_results_by_column_np.sum(axis=0) > 0).sum()} column(s) with at least one exception.")

    def __update_results_by_column(self, results_col, original_cols):
        rows_with_issue = np.where(results_col)
        score_increment = 1 / len(original_cols)
        original_col_idxs = [np.where(self.orig_df.columns == c) for c in original_cols]
        self.test_results_by_column_np[rows_with_issue, original_col_idxs] = \
            self.test_results_by_column_np[rows_with_issue, original_col_idxs] + score_increment

    def __process_analysis_binary(
            self,
            test_id,
            col_name,
            original_cols,
            test_series,
            pattern_string,
            exception_str="",
            allow_patterns=True,
            display_info=None):
        """
        Used by tests that produce a binary column indicating the test result.
        If all values in the column match the test, we update self.patterns_df.
        If most, but not all rows produce a positive result, we flag any exceptions. In this case, we add a results
        column and update self.results_summary_arr.

        test_series is the result of running the test. If the pattern consistently holds, this will be all True. If
        there are exceptions, the rows where there are exceptions will have value False, and  will be flagged in the
        results dataframe, which will be the opposite rows; the results dataframe contains True where there are
        exceptions.

        If allow_patterns is False, this will not update the patterns_arr. This is used for tests which only
        identify exceptions and not patterns, such as LARGE_GAPS
        """
        test_series = np.array(test_series)

        num_true = test_series.tolist().count(True)
        if allow_patterns and num_true == self.num_rows:
            self.patterns_arr.append([test_id, col_name, pattern_string, display_info])

        if (self.num_rows - self.contamination_level) <= num_true < self.num_rows:
            if allow_patterns:
                summary_str = f'{pattern_string}, with exceptions {exception_str}'
            else:
                summary_str = f'{pattern_string}, {exception_str}'
            summary_str = summary_str.replace(" .", ".").replace("..",".").replace(".,",",")

            # Update results_summary_arr
            self.results_summary_arr.append([
                test_id,
                col_name,
                summary_str,
                self.num_rows - num_true,
                display_info])

            # Update results_arr
            results_col_name = self.get_results_col_name(test_id, col_name)
            results_col = ~test_series.astype(bool)
            self.__add_result_column(results_col_name, results_col, original_cols)

            # Update test_results_by_column_df
            # todo: call this for all tests that don't call process_binary or process_counts
            #   I think only a few others do so far
            self.__update_results_by_column(results_col, original_cols)

    def __process_analysis_counts(
            self,
            test_id,
            col_name,
            original_cols,
            test_series,
            pattern_string_1,
            pattern_string_2):
        """
        Used by tests that produce a count column indicating the test result.
        test_series is the result of running the test. If the pattern consistently holds, this will contain a small
        number of values, each fairly frequent. There are exceptions if there are a small number of counts, but some
        values are much less common than the others.
        """

        def arr_to_str(arr):
            s = ""
            sorted_arr = sorted(arr)
            for x_ix, x in enumerate(sorted_arr):
                s += str(x)
                if x_ix == len(sorted_arr)-1:
                    break
                elif len(sorted_arr) == 2 and x_ix == 0:
                    s += " or "
                elif len(sorted_arr) > 2:
                    if x_ix == len(sorted_arr)-2:
                        s += ", or "
                    else:
                        s += ", "
            return s

        test_series = pd.Series(test_series)

        if test_series.nunique() == 1:
            self.patterns_arr.append([test_id, col_name, f'{pattern_string_1} {test_series[0]} {pattern_string_2}', None])
        elif test_series.nunique() <= 5:
            counts_series = test_series.value_counts(normalize=False)
            low_vals = [x for x, y in zip(counts_series.index, counts_series.values) if y < self.contamination_level]
            if len(low_vals) > 0:
                results_col = test_series.isin(low_vals)

                # Update results_summary_arr
                high_vals = [x for x in counts_series.index if x not in low_vals]
                self.results_summary_arr.append([
                    test_id,
                    col_name,
                    f'{pattern_string_1} {arr_to_str(high_vals)}{pattern_string_2}, with exceptions.',
                    results_col.tolist().count(True),
                    None  # Display information
                ])

                # Update results_arr
                results_col_name = self.get_results_col_name(test_id, col_name)
                self.__add_result_column(results_col_name, results_col, original_cols)

                # Update test_results_by_column_df
                self.__update_results_by_column(results_col, original_cols)

    def __output_current_test(self, test_num, test_id):
        """
        Executed as tests run to allow monitoring progress.
        """
        multiline_test_desc = wrap(self.test_dict[test_id][TEST_DEFN_DESC], 70)
        print(f"Executing test {test_num:3}: {test_id+':':<30} {multiline_test_desc[0]}")
        filler = ''.join([" "]*52)
        for line in multiline_test_desc[1:]:
            print(f'{filler} {line}')

    def __add_synthetic_column(self, col_name, col_values):
        """
        Add a column with the specified name and values to self.synth_df.
        This uses concat(), instead of simply adding columns, to avoid inefficiency issues.
        """
        self.synth_df = pd.concat([
            self.synth_df,
            pd.DataFrame({col_name: col_values})],
            axis=1)

    def __add_result_column(self, col_name, col_values, original_cols):
        """
        Add a column with the specified name and values to self.test_results_df.
        This uses concat(), instead of simply adding columns, to avoid inefficiency issues.
        """
        self.col_to_original_cols_dict[col_name] = original_cols

        self.test_results_df = pd.concat([
            self.test_results_df,
            pd.DataFrame({col_name: col_values})],
            axis=1)

    def check_columns_same_scale_2(self, col_name_1, col_name_2, order=2):
        med_1 = self.column_medians[col_name_1]
        med_2 = self.column_medians[col_name_2]
        if med_1 != 0 and med_2 != 0:
            ratio_1_2 = abs(med_1 / med_2)
            if ratio_1_2 < (1.0/order) or ratio_1_2 > order:
                return False
        return True

    def check_columns_same_scale_3(self, col_name_1, col_name_2, col_name_3, order=2):
        med_1 = self.column_medians[col_name_1]
        med_2 = self.column_medians[col_name_2]
        med_3 = self.column_medians[col_name_3]
        if med_1 != 0 and med_2 != 0:
            ratio_1_2 = abs(med_1 / med_2)
            if ratio_1_2 < (1/order) or ratio_1_2 > order:
                return False
        if med_1 != 0 and med_3 != 0:
            ratio_1_3 = abs(med_1 / med_3)
            if ratio_1_3 < (1/order) or ratio_1_3 > order:
                return False
        if med_2 != 0 and med_3 != 0:
            ratio_2_3 = abs(med_2 / med_3)
            if ratio_2_3 < (1/order) or ratio_2_3 > order:
                return False
        return True

    def check_results_for_null(self, test_series, col_name, subset):
        """
        Used by tests that work with any number of columns, and where Null values do not violate the general pattern.
        """
        if col_name:
            test_series = test_series | self.orig_df[col_name].isna()
        for col in subset:
            test_series = test_series | self.orig_df[col].isna()
        return test_series

    #########################################################################
    # Internal methods to get columns or sets of columns
    #########################################################################

    def __get_column_pairs_unique(self):
        """
        Returns a set of all pairs of columns A & B, other than where A == B. This will return both
        (A,B) and (B,A). This may be used to test, for example, where A >> B and B >> A.
        """
        pairs_arr = []
        for col_idx_1 in range(len(self.orig_df.columns)-1):
            col_name_1 = self.orig_df.columns[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.orig_df.columns)):
                col_name_2 = self.orig_df.columns[col_idx_2]
                pairs_arr.append((col_name_1, col_name_2))
        return pairs_arr

    def __get_binary_column_pairs(self, same_vocabulary=True):
        """
        """
        pairs_arr = []
        for col_name_1 in self.binary_cols:
            col_1_vals = set(self.orig_df[col_name_1].unique())
            for col_name_2 in self.binary_cols:
                col_2_vals = set(self.orig_df[col_name_2].unique())
                if col_name_1 == col_name_2:
                    continue
                if same_vocabulary and len(col_1_vals.intersection(col_2_vals)) != 2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return pairs_arr

    def __get_binary_column_pairs_unique(self, same_vocabulary=True):
        """
        if same_vocabulary is True, this returns only pairs of binary columns that contain the same 2 values.
        """
        pairs_arr = []
        for col_idx_1 in range(len(self.binary_cols)-1):
            col_name_1 = self.binary_cols[col_idx_1]
            col_1_vals = set(self.column_unique_vals[col_name_1])
            for col_idx_2 in range(col_idx_1+1, len(self.binary_cols)):
                col_name_2 = self.binary_cols[col_idx_2]
                col_2_vals = set(self.column_unique_vals[col_name_2])
                if same_vocabulary and len(col_1_vals.intersection(col_2_vals)) != 2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return pairs_arr

    def __get_numeric_column_pairs(self):
        """
        This behaves the same as __get_column_pairs(), but returns only pairs where both columns are numeric.
        """
        num_pairs = len(self.numeric_cols) * len(self.numeric_cols)
        if num_pairs > self.max_combinations:
            return num_pairs, None

        pairs_arr = []
        for col_name_1 in self.numeric_cols:
            for col_name_2 in self.numeric_cols:
                if col_name_1 == col_name_2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def __get_numeric_column_pairs_unique(self):
        """
        Similar to __get_numeric_column_pairs(), but returns each unique pair; this will return (A,B), but not (B,A)
        """
        num_pairs = math.comb(len(self.numeric_cols), 2)
        if num_pairs > self.max_combinations:
            return num_pairs, None

        pairs_arr = []
        for col_idx_1 in range(len(self.numeric_cols)-1):
            col_name_1 = self.numeric_cols[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.numeric_cols)):
                col_name_2 = self.numeric_cols[col_idx_2]
                if col_name_1 == col_name_2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def __get_numeric_column_triples(self):
        num_triples = len(self.numeric_cols) * len(self.numeric_cols) * len(self.numeric_cols)
        if num_triples > self.max_combinations:
            return num_triples, None

        triples_arr = []
        for col_name_1 in self.numeric_cols:
            for col_name_2 in self.numeric_cols:
                if col_name_1 == col_name_2:
                    continue
                for col_name_3 in self.numeric_cols:
                    if (col_name_3 == col_name_1) or (col_name_3 == col_name_2):
                        continue
                    triples_arr.append((col_name_1, col_name_2, col_name_3))
        return num_triples, triples_arr

    def __get_numeric_column_triples_unique(self):
        num_triples = math.comb(len(self.numeric_cols), 3)
        if num_triples > self.max_combinations:
            return num_triples, None

        triples_arr = []
        for col_ix_1 in range(len(self.numeric_cols)-2):
            col_name_1 = self.numeric_cols[col_ix_1]
            for col_ix_2 in range(col_ix_1 + 1, len(self.numeric_cols)-1):
                col_name_2 = self.numeric_cols[col_ix_2]
                for col_ix_3 in range(col_ix_2+1, len(self.numeric_cols)):
                    col_name_3 = self.numeric_cols[col_ix_3]
                    triples_arr.append((col_name_1, col_name_2, col_name_3))
        return num_triples, triples_arr

    def __get_string_column_pairs_unique(self):
        pairs_arr = []
        for col_idx_1 in range(len(self.string_cols)-1):
            col_name_1 = self.string_cols[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.string_cols)):
                col_name_2 = self.string_cols[col_idx_2]
                pairs_arr.append((col_name_1, col_name_2))
        return pairs_arr

    def __get_string_column_pairs(self):
        """
        This behaves the same as __get_column_pairs(), but returns only pairs where both columns are string.
        """
        pairs_arr = []
        for col_name_1 in self.string_cols:
            for col_name_2 in self.string_cols:
                if col_name_1 == col_name_2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return pairs_arr

    def __get_date_column_pairs_unique(self):
        pairs_arr = []
        for col_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_idx_2]
                pairs_arr.append((col_name_1, col_name_2))
        return pairs_arr

    #########################################################################
    # Clear issues
    #########################################################################
    def clear_issues(self, row_number=None, test_id=None, col_name=None):
        """
        This may be used to iteratively clean the results until all issues are acknowledged or understood.
        """
        pass

    def clear_issues_by_id(self, ids: list):
        """
        This may be used to iteratively clean the results until all issues are acknowledged or understood.
        This removes a set of issues identified by id. The id corresponds to the row number in results_summary_df.
        This means each time check_data_quality() is executed, the results will have different ids.
        """
        pass

    def restore_issues(self):
        """
        This set all the discovered exceptions back to the state when the tests were last run. It does not
        re-execute any tests; it undoes any removing of exceptions that was done by calling clear_issues()
        """
        self.results_summary_df = self.safe_results_summary_df.copy()
        self.test_results_df = self.safe_test_results_df.copy()
        self.test_results_by_column_np = self.safe_test_results_by_column_np.copy()

    #########################################################################
    # Tune the contamination rate
    #########################################################################
    def test_contamination_level(self, contamination_levels_arr=None):
        """
        This may be used to help determine an appropriate contamination rate to set for the process. Patterns in the
        date will only be recognized by the tool as patterns if there are less than the specified contamination rate
        of exceptions. Exceptions to patterns can only be found if the patterns are first recognized.
        Setting the contamination rate to a small value will identify only strong exceptions, but
        may miss some interesting patterns in the data. Setting a higher value will expose more patterns, but will
        also generate some noise.

        This presents a set of 3 plots: the number of issues found, the number of rows flagged at least once, and the
        number of columns flagged at least once, based on the contamination rate. It also returns these counts as
        three arrays.
        """
        if contamination_levels_arr is None:
            contamination_levels_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        pass

    #########################################################################
    # Data consistency checks for single columns of any type
    #########################################################################
    def __generate_missing(self):
        """
        Patterns without exceptions: 'missing vals all' has consistently non-missing values.
        Patterns with exception: 'missing vals most' has consistently non-missing values, with the exception of a None.
        """
        self.__add_synthetic_column('missing vals rand',
                                    [random.choice(['a', 'b', 'c', None]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('missing vals all ',
                                    [random.choice(['a', 'b', 'c']) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('missing vals most',
                                    [random.choice(['a', 'b', 'c']) for _ in range(self.num_synth_rows-2)] + [None] + [np.NaN])
        self.__add_synthetic_column('missing vals most null',
                                    [None] * (self.num_synth_rows-1) + ['a'])

    def __check_missing(self, test_id):
        for col_name in self.orig_df.columns:
            null_arr = self.orig_df[col_name].isna()
            non_null_arr = ~null_arr
            num_null = null_arr.sum()
            if (self.num_rows - self.contamination_level) < num_null < self.num_rows:
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    null_arr,
                    f"The column contains values that are consistently NULL"
                )
            if 0 <= num_null < self.contamination_level:
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    non_null_arr,
                    f"The column contains values that are consistently non-NULL"
                )

    def __generate_rare_values(self):
        """
        Patterns without exceptions: 'rare vals all' has consistently frequent values.
        Patterns with exception: 'rare vals most' has consistently frequent values, with the exception of 'z', a rare
            value.
        """
        self.__add_synthetic_column('rare_vals rand', [random.choice(string.ascii_letters)
                                                       for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('rare_vals all ', [random.choice(['a', 'b', 'c'])
                                                       for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('rare_vals most', [random.choice(['a', 'b', 'c'])
                                                       for _ in range(self.num_synth_rows - 1)] + ['z'])

    def __check_rare_values(self, test_id):
        # todo: allow users to specify a set of common values, and just flag anything else. Or in the clear issues,
        #   allow to clear there.

        # Check all column types except binary, where there are often rare values.
        for col_name in self.numeric_cols + self.date_cols + self.string_cols:
            if self.orig_df[col_name].nunique() > math.log2(self.num_rows):
                continue

            counts_series = self.orig_df[col_name].value_counts(normalize=False, dropna=False)
            all_rare_vals = [x for x, y in zip(counts_series.index, counts_series.values) if y < self.contamination_level]
            non_null_rare_vals = [x for x in all_rare_vals if not is_missing(x)]
            # It is not possible to sort None or NaN values
            rare_vals = sorted(non_null_rare_vals)
            for v in all_rare_vals:
                if v not in rare_vals:
                    rare_vals.append(v)

            all_common_vals = [x for x in counts_series.index if x not in rare_vals]
            non_null_common_vals = [x for x in all_common_vals if not is_missing(x)]
            common_vals = sorted(non_null_common_vals)
            for v in all_common_vals:
                if v not in common_vals:
                    common_vals.append(v)

            test_series = ~self.orig_df[col_name].isin(rare_vals)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f"The column consistently contains a small set of common values: {common_vals}",
                f"The rare values: {rare_vals}"
            )

    def __generate_unique_values(self):
        """
        Patterns without exceptions: 'unique vals all' has consistently unique values.
        Patterns with exception: 'unique vals most' has consistently unique values, with the exception of 500, which
            appears twice.
        """
        self.__add_synthetic_column('unique_vals rand', [random.randint(0, 250) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('unique_vals all ', np.arange(0, self.num_synth_rows))
        self.__add_synthetic_column('unique_vals most', np.arange(0, self.num_synth_rows-1).tolist() + [500])

    def __check_unique_values(self, test_id):
        for col_name in self.orig_df.columns:
            # The values are not considered to be all unique if any Null values are present
            if self.orig_df[col_name].isna().sum() > 0:
                continue

            # Test on a sample first
            counts_arr = self.sample_df[col_name].value_counts()
            test_series = [True if counts_arr[x] == 1 else False for x in self.sample_df[col_name]]
            if test_series.count(False) > 1:
                continue

            counts_arr = self.orig_df[col_name].value_counts()
            repeated_vals = [x for x, y in zip(counts_arr.index, counts_arr.values) if y > 1]
            test_series = [True if counts_arr[x] == 1 else False for x in self.orig_df[col_name]]
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains values that are consistently unique",
                f"including {repeated_vals[:5]} ({len(repeated_vals)} identified repeated values)"
            )

    def __generate_prev_values_dt(self):
        """
        Patterns without exceptions:
            'pattern_history_df_str all_a' has a repeating pattern.
            'pattern_history_df_num all_a' has a repeating pattern.
        Patterns with exception:
            'pattern_history_df_str most_a' has a repeating pattern, with exceptions (a new value).
            'pattern_history_df_str most_b' has a repeating pattern, with exceptions (an existing value in the wrong spot).
            'pattern_history_df_num most_a' has a repeating pattern, with one exception
            'pattern_history_df_num most_b'  has a trend, with one exception
        Columns not flagged:
            'pattern_history_df_num all_b' has a trend, but it is no stronger than predicting the previous value
            'pattern_history_df_num most_b' has a trend, but it is no stronger than predicting the previous value
        """
        # Categorical test cases - repeating values
        self.__add_synthetic_column('pattern_history_df_str all_a', ['a', 'b', 'c', 'd'] * (self.num_synth_rows // 4))
        self.__add_synthetic_column('pattern_history_df_str most_a', ['a', 'b', 'c', 'd'] * (self.num_synth_rows // 4))
        self.synth_df.loc[999, 'pattern_history_df_str most_a'] = 'x'
        self.__add_synthetic_column('pattern_history_df_str most_b', ['a', 'b', 'c', 'd'] * (self.num_synth_rows // 4))
        self.synth_df.loc[999, 'pattern_history_df_str most_b'] = 'a'

        # Numeric test cases - repeating values
        self.__add_synthetic_column('pattern_history_df_num all_a', [10, 20, 30, 40] * (self.num_synth_rows // 4))
        self.__add_synthetic_column('pattern_history_df_num most_a', [10, 20, 30, 40] * (self.num_synth_rows // 4))
        self.synth_df.loc[999, 'pattern_history_df_num most_a'] = 10

        # Numeric test cases - trend
        # todo: the DT can't beat just predicting the previous value
        self.__add_synthetic_column('pattern_history_df_num all_b', np.arange(self.num_synth_rows))
        self.__add_synthetic_column('pattern_history_df_num most_b', np.arange(self.num_synth_rows))
        self.synth_df.loc[999, 'pattern_history_df_num most_b'] = 10

    def __check_prev_values_dt(self, test_id):
        """
        This test cannot test the first 10 rows, as the test requires 10 prior rows of history.
        """

        look_back_range = 10  # The number of previous values examined to predict the current value

        for col_name in self.orig_df.columns:
            df = pd.DataFrame()
            df[col_name] = self.orig_df[col_name]
            # todo: if we can create a good DT with 10 lags, see if can with less. Loop until use as few lags as possible.
            for i in range(1, look_back_range):
                df[f"Lag_{i}"] = df[col_name].shift(i)
            # For any simple DTs, we should be able to train on a fairly small set of rows
            df = df.sample(n=min(len(df), 1000))

            n_unique_vals = self.orig_df[col_name].nunique()

            x_train = df.drop(columns=[col_name])
            x_train = x_train[look_back_range:]
            y_train = df[col_name][look_back_range:]

            x_train = x_train.replace([np.inf, -np.inf, np.NaN, None], sys.maxsize)
            y_train = y_train.replace([np.inf, -np.inf, np.NaN, None], sys.maxsize)

            if col_name in self.string_cols or col_name in self.binary_cols or n_unique_vals <= look_back_range:
                # todo: skip the first 10 rows, as they have no history to predict based on
                if self.orig_df[col_name].nunique() > look_back_range:
                    continue

                # Do not flag rare values, as there is another test for that.
                # Determine the number of non-rare unique values, so we can set the capacity of the decision tree to
                # a reasonable size.
                rare_values = []
                for v in self.orig_df[col_name].unique():
                    if self.orig_df[col_name].tolist().count(v) < self.contamination_level:
                        rare_values.append(v)

                # Allow the DT to create a rule for each non-rare value. Each rule may include multiple features in
                # the decision path.
                clf = DecisionTreeClassifier(max_leaf_nodes=max(2, self.orig_df[col_name].nunique()))

                # I think can leave out -- we fill NaN above.
                # for c in x_train.columns:
                #     if c in self.numeric_cols:
                #         x_train[c] = x_train[c].replace([np.inf, -np.inf], x_train[c].median())
                #         x_train[c] = x_train[c].fillna(x_train[c].median())
                #     else:
                #         try:
                #             x_train[c] = x_train[c].replace([np.inf, -np.inf], x_train[c].mode()[0])
                #         except:
                #             p = 7
                #         x_train[c] = x_train[c].fillna(x_train[c].mode()[0])
                x_train = pd.get_dummies(x_train, columns=x_train.columns)

                # commented out, but we may wish to put the mode() back. though it did give warnings
                #
                # try:
                #     print()
                #     print(col_name)
                #     print(y_train.mode()[0])
                #     y_train = y_train.fillna(y_train.mode()[0])
                # except:
                #     p = 9

                # Ensure the Y column is in a consistent format, using ordinal values
                y_train_numeric = y_train.map({x: y for x, y in zip(y_train.unique(), range(y_train.nunique()))})
                y_train_reverse = {y: x for x, y in zip(y_train.unique(), range(y_train.nunique()))}
                clf.fit(x_train, y_train_numeric)
                # todo: predict on a sample first
                y_pred = clf.predict(x_train)
                f1 = f1_score(y_train_numeric, y_pred, average='micro') # We do not use macro, as some rare classes may have poor f1 scores even for good trees.
                if f1 < 0.9:
                    continue

                # Test simply predicting the most common value
                f1_naive = f1_score(y_train_numeric, [y_train_numeric.mode()[0]] * len(y_pred), average='macro')
                if f1_naive >= (f1 - 0.05):
                    continue

                # Test simply predicting the previous value
                y_prev = y_train_numeric.shift(1)
                f1_naive = f1_score(y_train_numeric[1:], y_prev[1:], average='macro') # Elment 0 will have NaN, so can not be evaluated.
                if f1_naive >= (f1 - 0.05):
                    continue

                test_series = [x == y or x in rare_values for x, y in zip(y_train_numeric, y_pred)]
                test_series = [True]*look_back_range + test_series
                rules = tree.export_text(clf)
            elif col_name in self.numeric_cols:  # todo: try to do date columns here too
                # Skip columns that have very little variance
                q1 = self.orig_df[col_name].quantile(0.25)
                med = self.orig_df[col_name].quantile(0.50)
                q3 = self.orig_df[col_name].quantile(0.75)
                if med != 0:
                    if ((q3 - q1) / self.orig_df[col_name].quantile(0.5)) < 0.5:
                        continue
                else:
                    if (q1 == med) or (q3 == med):
                        continue

                y_train_reverse = None
                # Get the normalized MAE when using a decision tree
                regr = DecisionTreeRegressor(max_leaf_nodes=look_back_range)
                regr.fit(x_train, y_train)
                # todo: predict on a sample first
                y_pred = regr.predict(x_train)
                mae = metrics.median_absolute_error(y_train, y_pred)
                if self.column_medians[col_name] != 0:
                    norm_mae = abs(mae / self.column_medians[col_name])
                else:
                    norm_mae = np.inf
                if norm_mae > 0.1:
                    continue

                # Get the normalized MAE when simply predicting the median
                naive_mae = metrics.median_absolute_error(y_train, [y_train.quantile(0.5)] * len(y_train))
                if mae > (naive_mae * 0.5):
                    continue

                # Test simply predicting the previous value
                y_prev = y_train.shift(1)
                # Element 0 will have NaN, so can not be evaluated.
                naive_mae = metrics.median_absolute_error(y_train[1:], y_prev[1:])
                if mae > (naive_mae * 0.5):
                    continue

                test_series = [True if y == 0 else (x/y) < (y / 10.0) for x, y in zip(y_train, y_pred)]
                test_series = [True]*look_back_range + test_series
                rules = tree.export_text(regr)
            else:
                continue  # todo: can remove once support date columns

            # The export of the rules has the column names in the format 'feature_1' and so on. We replace these with
            # the actual column names. These are cleaned further below.
            cols = []
            for c_idx, c_name in enumerate(x_train.columns):
                rule_col_name = f'feature_{c_idx} '
                if rule_col_name in rules:
                    cols.append(c_name)
                    rules = rules.replace(rule_col_name, c_name + ' ')

            # We map the numeric values in the target column back to their original values
            if y_train_reverse:
                for v in y_train_reverse.keys():
                    rules = rules.replace(f"class: {v}", f"value: {y_train_reverse[v]}")

            # We map the split points (in the form of Lag_1_[value] <= 0.50) to a more readable format
            if y_train_reverse is None:  # Regression
                for i in range(look_back_range):
                    rules = rules.replace(f"Lag_{i}", f"The value {i} rows previously")
            else:  # Classification
                for i in range(look_back_range):
                    for v in y_train_reverse.keys():
                        val = y_train_reverse[v]
                        rules = rules.replace(f"Lag_{i}_{val} <= 0.50", f"The value {i} rows previously was not '{val}'")
                        rules = rules.replace(f"Lag_{i}_{val} >  0.50", f"The value {i} rows previously was '{val}'")
                        # In some cases, '.0' is added to the end of the vals
                        rules = rules.replace(f"Lag_{i}_{val}.0 <= 0.50", f"The value {i} rows previously was not '{val}'")
                        rules = rules.replace(f"Lag_{i}_{val}.0 >  0.50", f"The value {i} rows previously was '{val}'")

            # Check the tree is not trivial in that it has the same prediction in each leaf (the most common value)
            # todo: add this check for regression as well
            rules_lines = rules.split('\n')
            tree_preds = set()
            for line in rules_lines:
                if "class: " in line:
                    tree_preds.add(line.split('class: ')[1])
            if len(tree_preds) == 1:
                continue

            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                np.array(test_series),
                (f"The values in {col_name} can consistently be predicted from the previous values in the column "
                 f"with a decision tree using the following rules: \n\n{rules}"),
                "")

    #########################################################################
    # Data consistency checks for pairs of columns of any type
    #########################################################################

    def __generate_matched_missing(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('matched_missing_vals rand_a', [random.choice(['a', 'b', 'c', None]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('matched_missing_vals rand_b', [random.choice(['a', 'b', 'c', None]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('matched_missing_vals all', self.synth_df['matched_missing_vals rand_a'])
        self.__add_synthetic_column('matched_missing_vals most', self.synth_df['matched_missing_vals rand_a'])
        if self.synth_df.loc[999, 'matched_missing_vals most'] is None:
            self.synth_df.loc[999, 'matched_missing_vals most'] = 'a'
        else:
            self.synth_df.loc[999, 'matched_missing_vals most'] = None

    def __check_matched_missing(self, test_id):
        for col_idx_1 in range(len(self.orig_df.columns)-1):
            col_name_1 = self.orig_df.columns[col_idx_1]
            col_1_missing_arr = [is_missing(x) for x in self.orig_df[col_name_1]]
            num_missing_1 = col_1_missing_arr.count(True)
            if self.verbose >= 2 and col_idx_1 > 0 and col_idx_1 % 25 == 0:
                print(f"  Examining column {col_idx_1} of {len(self.orig_df.columns)} columns")
            if num_missing_1 < self.contamination_level:
                continue
            if num_missing_1 > (self.num_rows - self.contamination_level):
                continue
            for col_idx_2 in range(col_idx_1 + 1, len(self.orig_df.columns)):
                col_name_2 = self.orig_df.columns[col_idx_2]
                col_2_missing_arr = [is_missing(x) for x in self.orig_df[col_name_2]]
                num_missing_2 = col_2_missing_arr.count(True)
                if num_missing_2 < self.contamination_level:
                    continue
                if num_missing_2 > (self.num_rows - self.contamination_level):
                    continue

                # If the difference between the number of missing values is too large, there is not a pattern
                if abs(num_missing_1 - num_missing_2) > self.contamination_level:
                    continue

                test_series = [x == y for x, y in zip(col_1_missing_arr, col_2_missing_arr)]

                # Determine if there is already a pattern found, which these columns are part of. If so, simply add
                # the new column to the pattern.
                found_existing_pattern = False
                if test_series.count(False) == 0:
                    for pattern_idx, existing_pattern in enumerate(self.patterns_arr):
                        if existing_pattern[0] != test_id:
                            continue
                        if col_name_1 in existing_pattern[1] or col_name_2 in existing_pattern[1]:
                            if col_name_1 not in existing_pattern[1]:
                                existing_pattern[1] += " AND " + col_name_1
                            if col_name_2 not in existing_pattern[1]:
                                existing_pattern[1] += " AND " + col_name_2
                            existing_pattern[2] = (f'The columns consistently have missing values in the same rows, '
                                                   f'with {num_missing_1} missing values.')
                            self.patterns_arr[pattern_idx] = existing_pattern
                            found_existing_pattern = True
                            break
                if found_existing_pattern:
                    continue

                self.__process_analysis_binary(
                    test_id,
                    f"{col_name_1} AND {col_name_2}",
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The columns "{col_name_1}" (with {num_missing_1} Null values) and "{col_name_2}" (with '
                     f'{num_missing_2} Null values) consistently have missing values in the same rows.')
                )

    def __generate_same(self):
        self.__add_synthetic_column('same rand', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same all ', self.synth_df['same rand'])
        self.__add_synthetic_column('same most', self.synth_df['same rand'])
        self.synth_df.loc[999, 'same most'] = 2.1

    def __check_same(self, test_id):
        def test_arrs(arr1, arr2, is_sample):
            if arr1.isna().sum() > (len(arr1) * 0.75):
                return False
            if arr2.isna().sum() > (len(arr2) * 0.75):
                return False

            test_series = [x == y or is_missing(x) or is_missing(y) for x, y in zip(arr1, arr2)]
            if is_sample:
                return test_series.count(False) < 1
            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f'The values in "{col_name_2}" are consistently the same as those in "{col_name_1}"')

        def test_pair():
            if not test_arrs(self.sample_df[col_name_1], self.sample_df[col_name_2], is_sample=True):
                return
            test_arrs(self.orig_df[col_name_1], self.orig_df[col_name_2], is_sample=False)

        num_pairs, pairs = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
               f"max_combinations is currently set to {self.max_combinations:,}"))
        else:
            for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
                if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                    print(f"  Processing pair number {pair_idx:,} of {len(pairs):,} pairs of numeric columns")
                test_pair()

        # todo: add checks for size for other column types as did for numeric
        for col_name_1, col_name_2 in self.__get_binary_column_pairs_unique():
            test_pair()

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            test_pair()

        for col_name_1, col_name_2 in self.__get_date_column_pairs_unique():
            test_pair()

    def __generate_same_or_constant(self):
        """
        Patterns without exceptions: 'same_or_const all' consistently has either the same value as 'same_or_const rand'
            or a constant value, 34.5
        Patterns with exception: 'same_or_const most' usually has either the same value as 'same_or_const rand'
            or a constant value, 34.5, with the exception of row 999
        """
        self.__add_synthetic_column('same_or_const rand', [random.random() - 0.5 for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same_or_const all ', [x if y % 2 == 0 else 34.5
            for x, y in zip(self.synth_df['same_or_const rand'], range(self.num_synth_rows))])
        self.__add_synthetic_column('same_or_const most', self.synth_df['same_or_const all '].copy())
        self.synth_df.loc[999, 'same_or_const most'] = 2.1

    def __check_same_or_constant(self, test_id):
        """
        For each pair of columns we check both directions, if A is either the same as B or a small set of other values,
        or if B is either the same as A or a small set of other values.
        """

        def test_arrs(arr1, arr2, is_sample):
            """
            This may be run on either a sample of the full data or the full data. If run on a sample, we check if the
            two arrays appear to exhibit the pattern and return True if so and False otherwise. If True, this will
            be called again on the full dataset, which may save a pattern or exception if the pattern is true for
            the majority of rows.
            """
            if arr1.isna().sum() > (len(arr1) * 0.75):
                return False
            if arr2.isna().sum() > (len(arr2) * 0.75):
                return False

            same_indicator = [x == y or is_missing(x) or is_missing(y) for x, y in zip(arr1, arr2)]

            # Exclude column pairs which are not the same value in at least 10% of rows
            if same_indicator.count(True) < (len(arr1) * 0.1):
                return False

            # Exclude column pairs which are the same value in over 95% of rows
            if same_indicator.count(True) > (len(arr1) * 0.95):
                return False

            other_values_1 = pd.Series([np.NaN if y == 1 else x for x, y in zip(arr1, same_indicator)])
            other_values_2 = pd.Series([np.NaN if y == 1 else x for x, y in zip(arr2, same_indicator)])
            vc_1 = other_values_1.value_counts()
            vc_2 = other_values_2.value_counts()
            if len(vc_1) <= 5:
                common_alternatives = [x for x, y in zip(vc_1.index, vc_1.values) if y > self.contamination_level]
                col_values = np.array([True if ((x == 1) or (y in common_alternatives)) else False
                                       for x, y in zip(same_indicator, arr1)])
                if is_sample:
                    return col_values.tolist().count(False) <= 1
                else:
                    self.__process_analysis_binary(
                        test_id,
                        f'"{col_name_2}" AND "{col_name_1}"',
                        [col_name_2, col_name_1],
                        col_values,
                        (f'The values in "{col_name_1}" are consistently either the same as those in "{col_name_2}", '
                         f'or one of {common_alternatives}'))
            elif len(vc_2) <= 5:
                common_alternatives = [x for x, y in zip(vc_2.index, vc_2.values) if y > self.contamination_level]
                col_values = np.array([True if ((x == 1) or (y in common_alternatives)) else False
                                       for x, y in zip(same_indicator, arr2)])
                if is_sample:
                    return col_values.tolist().count(False) <= 1
                else:
                    self.__process_analysis_binary(
                        test_id,
                        f'"{col_name_1}" AND "{col_name_2}"',
                        [col_name_1, col_name_2],
                        col_values,
                        (f'The values in "{col_name_2}" are consistently either the same as those in "{col_name_1}, '
                         f'or one of {common_alternatives}'))
            else:
                return False

        # For this test, we do not use self.sample_df, as it has the extreme values removed, which may be preferable
        # to keep for this test
        sample_df = self.orig_df.sample(n=min(len(self.orig_df), 50))

        col_pairs = self.__get_column_pairs_unique()
        if len(col_pairs) > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {len(col_pairs):,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(col_pairs):,} combinations.")

            # Skip binary columns
            if col_name_1 in self.binary_cols or col_name_2 in self.binary_cols:
                continue

            # Skip cases where one column is almost entirely a single value.
            vc = self.orig_df[col_name_1].value_counts(normalize=True)
            if vc.sort_values(ascending=False).values[0] > 0.99:
                continue
            vc = self.orig_df[col_name_2].value_counts(normalize=True)
            if vc.sort_values(ascending=False).values[0] > 0.99:
                continue

            if not test_arrs(sample_df[col_name_1], sample_df[col_name_2], is_sample=True):
                continue
            test_arrs(self.orig_df[col_name_1], self.orig_df[col_name_2], is_sample=False)

    #########################################################################
    # Data consistency checks for single numeric columns
    #########################################################################

    # All __generate_xxx methods follow the pattern of generating three columns: one with no pattern, one with a pattern
    # but no exceptions, and one with a pattern but 1 exception, in the final row.

    def __generate_positive_values(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('positive rand', [random.random() - 0.5 for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('positive all ', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('positive most', [random.random() for _ in range(self.num_synth_rows-1)] + [-0.5])

    def __check_positive_values(self, test_id):
        for col_name in self.numeric_cols:
            test_series = (self.orig_df[col_name].isna()) | (self.orig_df[col_name] >= 0)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains consistently positive values")

    def __generate_negative_values(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('negative rand', [-random.random() + 0.5 for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('negative all ', [-random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('negative most', [-random.random() for _ in range(self.num_synth_rows-1)] + [0.5])

    def __check_negative_values(self, test_id):
        for col_name in self.numeric_cols:
            test_series = (self.orig_df[col_name].isna()) | (self.orig_df[col_name] <= 0)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains consistently negative values")

    def __generate_number_decimals(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('num decimals rand', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('num decimals all ', [round(random.random(), 2) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('num decimals most', [round(random.random(), 2) for _ in range(self.num_synth_rows-1)] + [0.5665])

    def __check_number_decimals(self, test_id):
        for col_name in self.numeric_cols:

            # Test first on a sample
            num_digits_series = self.sample_df[col_name].apply(get_num_digits)
            num_digits_non_null_series = pd.Series([get_num_digits(x) for x in self.sample_df[col_name] if not is_missing(x)])

            counts_series = num_digits_non_null_series.value_counts(normalize=False)
            most_common_num_digits = counts_series.sort_values().index[-1]
            rare_num_digits = [x for x in counts_series.index if (x > (most_common_num_digits * 1.2)) and (x >= (most_common_num_digits + 2))]

            test_series = [x not in rare_num_digits for x in num_digits_series]
            if test_series.count(False) > 1:
                continue

            # Test on the full column
            num_digits_series = self.orig_df[col_name].apply(get_num_digits)
            num_digits_non_null_series = pd.Series([get_num_digits(x) for x in self.orig_df[col_name] if not is_missing(x)])

            counts_series = num_digits_non_null_series.value_counts(normalize=False)
            most_common_num_digits = counts_series.sort_values().index[-1]
            rare_num_digits = [x for x in counts_series.index if (x > (most_common_num_digits * 1.2)) and (x >= (most_common_num_digits + 2))]

            # An alternative formulation for rare numbers of digits
            # [x for x, y in zip(counts_series.sort_values(ascending=True).index, counts_series.sort_values(ascending=True).values) if y <= self.contamination_level]

            common_num_digits = [x for x in counts_series.index if x <= most_common_num_digits]
            test_series = [x not in rare_num_digits for x in num_digits_series]

            # We just flag values that have significantly more decimals than normal. Where values have less
            # than normal, this may simply be that there are zeros in the least significant positions.
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The column contains values consistently with {array_to_str(common_num_digits)} decimals or less',
                allow_patterns=((len(common_num_digits) > 1) or (common_num_digits[0] != 0))
            )

    def __generate_column_increasing(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('col_asc rand', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('col_asc all', sorted([round(random.random(), 2) for _ in range(self.num_synth_rows)]))
        self.__add_synthetic_column('col_asc most', sorted([round(random.random(), 2) for _ in range(self.num_synth_rows-1)]) + [0.0001])
        # todo: add date example

    def __check_column_increasing(self, test_id):
        for col_name in self.numeric_cols + self.date_cols:
            if col_name in self.numeric_cols:
                test_series = (self.orig_df[col_name].diff() >= 0) | [is_missing(x) for x in self.orig_df[col_name].diff()]
            else:
                test_series = np.array([x.total_seconds() for x in (self.orig_df[col_name] - self.orig_df[col_name].shift())]) >= 0
                test_series = test_series | self.orig_df[col_name].diff().isna()
            # The shift operation is undefined for the first row, which results in a NaN that we fill here.
            test_series[0] = True
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains consistently ascending values")

    def __generate_column_decreasing(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('col desc 1', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('col desc 2', sorted([round(random.random(), 2) for _ in range(self.num_synth_rows)], reverse=True))
        self.__add_synthetic_column('col desc 3', sorted([round(random.random(), 2) for _ in range(self.num_synth_rows-1)], reverse=True) + [5.1])
        # todo: add date column to demo data

    def __check_column_decreasing(self, test_id):
        for col_name in self.numeric_cols + self.date_cols:
            if col_name in self.numeric_cols:
                test_series = (self.orig_df[col_name].diff() <= 0) | [is_missing(x) for x in self.orig_df[col_name].diff()]
            else:
                test_series = np.array([x.total_seconds() for x in (self.orig_df[col_name] - self.orig_df[col_name].shift())]) <= 0
                test_series = test_series | self.orig_df[col_name].diff().isna()
            # The shift operation is undefined for the first row, which results in a NaN we fill here.
            test_series[0] = True
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains consistently descending values")

    def __generate_column_tends_asc(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('tends_asc rand', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('tends_asc all ', [x + random.random() for x in range(self.num_synth_rows)])
        self.__add_synthetic_column('tends_asc most', [x + random.random() for x in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'tends_asc most'] = 4

    def __check_column_tends_asc(self, test_id):
        """
        Check if the values in the column are correlated with the row numbers. Identify any outliers.
        """
        row_numbers = list(range(self.num_rows))
        row_numbers_percentiles = pd.Series(row_numbers).rank(pct=True)
        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique(dropna=True) <= 2:
                continue
            if col_name in self.numeric_cols:
                spearman_corr = abs(self.orig_df[col_name].corr(pd.Series(row_numbers), method='spearman'))
            else:
                col_vals = [x.timestamp() for x in self.orig_df[col_name]]
                spearman_corr = abs(pd.Series(col_vals).corr(pd.Series(row_numbers), method='spearman'))
            if spearman_corr >= 0.95:
                col_percentiles = self.orig_df[col_name].rank(pct=True)

                # Test for positive correlation
                test_series = np.array([(abs(x-y) < 0.1) or is_missing(x) for x, y in zip(col_percentiles, row_numbers_percentiles)])
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    test_series,
                    f'"{col_name}" is consistently similar, with regards to percentile, to the row number')

    def __generate_column_tends_desc(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('tends_desc rand', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('tends_desc all ', [x + random.random() for x in range(self.num_synth_rows-1, -1, -1)])
        self.__add_synthetic_column('tends_desc most', [x + random.random() for x in range(self.num_synth_rows-1, -1, -1)])
        self.synth_df.loc[999, 'tends_desc most'] = 1000

    def __check_column_tends_desc(self, test_id):
        """
        Check if the values in the column are inversely correlated with the row numbers. Identify any outliers.
        """
        row_numbers = list(range(self.num_rows))
        row_numbers_percentiles = pd.Series(row_numbers).rank(pct=True)
        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique(dropna=True) <= 2:
                continue
            if col_name in self.numeric_cols:
                spearancorr = abs(self.orig_df[col_name].corr(pd.Series(row_numbers), method='spearman'))
            else:
                col_vals = [x.timestamp() for x in self.orig_df[col_name]]
                spearman_corr = abs(pd.Series(col_vals).corr(pd.Series(row_numbers), method='spearman'))
            if spearancorr >= 0.95:
                col_percentiles = self.orig_df[col_name].rank(pct=True)

                # Test for negative correlation
                test_series = np.array([(abs(x-(1.0 - y)) < 0.1) or is_missing(x) for x, y in
                                        zip(col_percentiles, row_numbers_percentiles)])
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    test_series,
                    f'"{col_name}" is consistently inversely similar, with regards to percentile, to the row number')

    def __generate_similar_previous(self):
        """
        Patterns without exceptions:
            'sim_prev_all' has values consistently similar to the previous values
        Patterns with exception:
            'sim_prev_most' has values consistently similar to the previous values, with 1 exception
        Not Flagged:
            'sim_prev_rand' has no relationship to the previous values
        """
        random_walk = [10.0]
        prev_val = random_walk[0]
        for i in range(self.num_synth_rows-1):
            new_val = prev_val + ((random.random() - 0.5) * 2.0)
            random_walk.append(new_val)
            prev_val = new_val

        self.__add_synthetic_column('sim_prev_rand', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim_prev_all', random_walk)
        self.__add_synthetic_column('sim_prev_most', random_walk)
        self.synth_df.loc[999, 'sim_prev_most'] += 15

    def __check_similar_previous(self, test_id):
        """
        Check only the previous value in the column. The test cannot be performed on the first row of each column.
        """
        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique(dropna=True) <= 2:
                continue
            diff_to_prev_arr = abs(self.orig_df[col_name].diff())
            diff_to_median_arr = abs(self.orig_df[col_name] - self.orig_df[col_name].median())
            test_series = diff_to_prev_arr < diff_to_median_arr
            test_series[0] = True  # Row 0 has no difference from the previous, so can not be tested

            # Test a reasonable number of values are closer to the previous value than to the median. It does not
            # have to be almost all, as many values may be close to the median as well.
            if test_series.tolist().count(True) < (self.num_rows * 0.75):
                continue

            # Test that those values not closer to the previous value, are still close
            q1 = self.orig_df[col_name].quantile(0.25)
            q3 = self.orig_df[col_name].quantile(0.75)
            iqr = abs(q3 - q1)
            test_series = diff_to_prev_arr < iqr
            test_series[0] = True
            test_series = test_series | self.orig_df[col_name].isna() | self.orig_df[col_name].diff().isna()

            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f'The values in "{col_name}" are consistently similar to the previous value, more so than they are '
                 f'similar to the median value of the column ({self.orig_df[col_name].median()})')
            )

    def __generate_unusual_numeric(self):
        """
        Patterns without exceptions: 'unusual_number rand' has values spanning many orders of magnitude and no pattern.
            'unusual_number all ' has a strong pattern, as all values are within 1 order of magnitude.
        Patterns with exception: 'unusual_number most' has a strong pattern, with most values having a value within an
            order of magnitude of each other, with one value many orders of magnitude larger. 'unusual_number rand'
            may occasionally randomly generate unsual values as well, which may be flagged as exceptions.
        """
        self.__add_synthetic_column('unusual_number rand', [random.randint(1000, 1_000_000)
                                                            for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('unusual_number all ', [random.randint(1, 5) * 100
                                                            for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('unusual_number most', [random.randint(1, 5) * 100
                                                            for _ in range(self.num_synth_rows-1)] + [999_999_999])

    def __check_unusual_numeric(self, test_id):
        for col_name in self.numeric_cols:
            # We consider only columns that do not contain fractions
            if abs(self.orig_df[col_name]).min() < 1:
                continue
            vals_arr = self.orig_df[col_name].replace([0, np.inf, -np.inf, np.NaN], self.orig_df[col_name].median())
            test_series = np.where(vals_arr > 0, np.log10(abs(vals_arr)).astype(int), -np.log10(abs(vals_arr))).astype(int)
            test_series = [math.pow(10, x) for x in test_series]
            self.__process_analysis_counts(
                test_id,
                col_name,
                [col_name],
                test_series,
                # todo: make this more clear. give min, max, mean values.
                (f"The column contains values in the range {self.orig_df[col_name].min()} to "
                 f"{self.orig_df[col_name].max()}, consistently in the order of"),
                 # f"{[math.pow(10, x) for x in sorted(pd.Series(test_series).unique())]}, that is: of 10 to the power "
                 # f"of"),
                "")

    def __generate_few_neighbors(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('few neighbors rand', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('few neighbors all ', [random.randint(1000, 2000) for _ in range(self.num_synth_rows-100)] + [random.randint(10_000, 20_000) for _ in range(100)])
        self.__add_synthetic_column('few neighbors most', [random.randint(1000, 2000) for _ in range(self.num_synth_rows-100)] + [random.randint(10_000, 20_000) for _ in range(99)] + [5000])

    def __check_few_neighbors(self, test_id):
        # This test does not identify patterns, but does identify rows that are distant from both the nearest point
        # before and point afterwards. This does not flag extreme values, only internal outliers.

        for col_name in self.numeric_cols:
            sorted_vals = copy.copy(self.orig_df[col_name].values)
            sorted_vals.sort()
            sorted_vals = pd.Series(sorted_vals)
            diff_from_prev = sorted_vals.diff(1)
            diff_from_next = sorted_vals.diff(-1)
            diff_threshold = (self.orig_df[col_name].max() - self.orig_df[col_name].min()) / 10.0

            test_arr = [True if not math.isnan(x) and not math.isnan(y) and (abs(x) > diff_threshold) and (abs(y) > diff_threshold)
                           else False for x, y in zip(diff_from_prev, diff_from_next)]
            num_isolated_points = test_arr.count(True)

            if num_isolated_points > 0:
                # Flag the correct rows. We currently have their indexes based on a sorted array.
                vals_arr = [x for x, y in zip(sorted_vals, test_arr) if y == True]
                test_series = [False if x in vals_arr else True for x in self.orig_df[col_name].values]
                # test_series = [not x for x in test_arr]
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    test_series,
                    (f"The test marked any values more than {diff_threshold:.3f} away from both the next smallest and "
                     "next largest values in the column"),
                     allow_patterns=False
                )

    def __generate_few_within_range(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('few in range rand', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'few in range all ',
            [random.randint(1000, 2000) for _ in range(self.num_synth_rows-100)] + [random.randint(10_000, 20_000)
                                                                                    for _ in range(100)])
        self.__add_synthetic_column(
            'few in range most',
            [random.randint(1000, 2000) for _ in range(self.num_synth_rows-100)] + [random.randint(10_000, 20_000)
                                                                                    for _ in range(97)] + [5000, 5001, 5002])

    def __check_few_within_range(self, test_id):
        # This identifies only internal outliers, and not extreme values. It flags values where there are few values
        # within a range on either side. To do this efficiently, it divides the column into 30 equal-width bins.
        # It is concerned with bins of width 1/10 the full data range, but to do this in a more robust manner, it
        # uses 30 bins, and considers sets of 3 bins.

        num_bins = 30
        for col_name in self.numeric_cols:
            if self.orig_df[col_name].nunique() < num_bins:
                continue

            bins = [-np.inf]
            min_val = self.orig_df[col_name].min()
            max_val = self.orig_df[col_name].max()
            col_range = max_val - min_val
            bin_width = col_range / num_bins
            for i in range(num_bins):
                bins.append(min_val + (i * bin_width))
            bins.append(np.inf)
            bin_labels = [int(x) for x in range(len(bins)-1)]
            binned_values = pd.cut(self.orig_df[col_name], bins, labels=bin_labels)
            bin_counts = binned_values.value_counts()

            rare_bins = []
            for bin_id in bin_labels[1:-1]:
                rows_count = bin_counts.loc[bin_id-1] + bin_counts.loc[bin_id] + bin_counts.loc[bin_id+1]
                if rows_count < self.contamination_level:
                    rare_bins.append(bin_id)
            if len(rare_bins) == 0:
                continue
            test_series = np.array([x not in rare_bins for x in binned_values])

            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently contains values that have several similar values in the column",
                (f"-- any values with fewer than an average of {math.floor(self.contamination_level)} neighbors within "
                 f"their and the neigboring bins (width {bin_width:.4f})"))

    def __generate_very_small(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('very small rand', [random.randint(1, 100_000_000)
                                                        for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('very small all ', [random.randint(1000, 2000)
                                                        for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('very small most', [random.randint(1000, 2000)
                                                        for _ in range(self.num_synth_rows-1)] + [3])

    def __check_very_small(self, test_id):
        """
        This uses inter-decile range.
        """

        for col_name in self.numeric_cols:
            d1 = self.orig_df[col_name].quantile(0.1)
            d9 = self.orig_df[col_name].quantile(0.9)
            lower_limit = d1 - (self.idr_limit * (d9 - d1))
            test_series = self.orig_df[col_name].isna() | (self.orig_df[col_name] > lower_limit)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f"The test marked any values less than {lower_limit} as very small given the 10th decile is {d1} and "
                 f"the 90th is {d9}. The coefficient is set at {self.idr_limit}"),
                allow_patterns=False
            )

    def __generate_very_large(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('very large 1', [random.randint(1, 100_000_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('very large 2', [random.randint(1000, 2000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('very large 3', [random.randint(1000, 2000) for _ in range(self.num_synth_rows-1)] + [30_000])

    def __check_very_large(self, test_id):
        for col_name in self.numeric_cols:
            q1 = self.orig_df[col_name].quantile(0.25)
            q3 = self.orig_df[col_name].quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))  # Using a stricter threshold than the 2.2 normally used
            test_series = self.orig_df[col_name].isna() | (self.orig_df[col_name] < upper_limit)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f"The test marked any values larger than {upper_limit:.2f} as very large given the 25th quartile "
                 f"is {q1} and 75th is {q3}"),
                allow_patterns=False
            )

    def __generate_very_small_abs(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('very_small_abs rand',
                                    [random.randint(1, 100_000_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('very_small_abs all',
                                    [random.randint(5000, 10000) for _ in range(self.num_synth_rows // 2)]
                                    + [random.randint(-10000, -5000)
                                       for _ in range(self.num_synth_rows - (self.num_synth_rows // 2))])
        self.__add_synthetic_column('very_small_abs most', self.synth_df['very_small_abs all'])
        self.synth_df.loc[999, 'very_small_abs most'] = 3

    def __check_very_small_abs(self, test_id):
        """
        Detect where values are unusually close to zero, where a feature has a significant number of both positive
        and negative values. Note, it is not necessary to check for very large absolute values, as these would be caught
        by the tests for very small and very large values. This test catches internal outliers, which are closer than
        normal to zero.
        """
        for col_name in self.numeric_cols:
            num_pos = (self.orig_df[col_name] > 0).tolist().count(True)
            num_neg = (self.orig_df[col_name] < 0).tolist().count(True)
            if num_pos < (self.num_rows * 0.10) or num_neg < (self.num_rows * 0.10):
                continue
            abs_vals = self.orig_df[col_name].abs()
            d1 = abs_vals.quantile(0.1)
            d9 = abs_vals.quantile(0.9)
            lower_limit = d1 - (self.idr_limit * (d9 - d1))
            test_series = self.orig_df[col_name].isna() | (abs_vals > lower_limit)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f"Some values were unusually close to zero. Any values with absolute value less than {lower_limit} "
                 f"were flagged, given the 10th percentile of absolute values is {d1} and the 90th is {d9}. " 
                 f" The coefficient is set at {self.idr_limit}"),
                "",
                allow_patterns=False
            )

    def __generate_multiple_constant(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('constant multiple rand',
                                    [random.randint(1, 100_000_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('constant multiple all',
                                    [random.randint(1, 2000) * 13.3 for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('constant multiple most',
                                    [random.randint(1, 2000) * 13.3 for _ in range(self.num_synth_rows-1)] + [30_000])

    def __check_multiple_constant(self, test_id):
        # todo: it is also possible to look at the gaps between values to find the common denomimator
        # todo: this considers only constants greater than 1.0, but in some cases, constants less than 1.0 may be
        # meaningful. Eg a constant of .25 or .5 is meaningless if the values are all integers, but is meaningful
        # if most values have decimals.

        def test_divisor(col_name, v):
            test_series = [math.isclose(x, y) or is_missing(x) or is_missing(y)
                           for x, y in zip(round(self.orig_df[col_name] / v), self.orig_df[col_name] / v)]
            n_multiples = test_series.count(True)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f"The column contains values that are consistently multiples of {v}",
                display_info={"value": v}
            )
            if n_multiples > (self.num_rows - self.contamination_level):
                return True
            return False

        exclude_list = [0, 1, -1]
        for col_name in self.numeric_cols:

            # Ensure there are many unique values in the column. Otherwise they may be multiples of a common
            # value in a trivial way.
            if self.orig_df[col_name].nunique() < (self.num_rows / 10):
                continue

            # Test if the rows tend to be multiples of any of the 5 smallest values. We test more than 1, as the
            # small values may themselves be the outliers.
            min_vals = [x for x in self.orig_df[col_name].unique() if x > 0.5 and x not in exclude_list]
            if len(min_vals) == 0:
                continue
            min_vals = sorted(list(set(min_vals)))[:5]
            found = False
            for v in min_vals:
                if test_divisor(col_name, v):
                    found = True
                    break

            # It may be that all records are several times the common multiple and the common multiple itself is not
            # in the data. Check if the smallest value available is any multiple up to 10 of the common multiple.
            if found:
                continue
            for v_div in range(2, 11):
                v = min_vals[0] / v_div
                if v > 0.5 and v not in exclude_list and test_divisor(col_name, v):
                    break

    def __generate_rounding(self):
        """
        Patterns without exceptions: 'rounding 1' and 'rounding 2' both have sets of values with similar numbers of
            rounded values.
        Patterns with exception: 'rounding 3' has one value with significantly more zeros. 'rounding 4' does as well,
            and is an example with a float column.
        """
        self.__add_synthetic_column('rounding 1', [random.randint(1, 1_000_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('rounding 2', [int(random.randint(1, 100) * (math.pow(10, random.randint(1, 4))))
                                                   for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('rounding 3', [int(random.randint(1, 100) * (math.pow(10, random.randint(1, 4))))
                                                   for _ in range(self.num_synth_rows-1)] + [10_000_000])
        self.__add_synthetic_column('rounding 4', [(random.randint(1, 100) * 0.25 * (math.pow(10, random.randint(1, 4))))
                                                   for _ in range(self.num_synth_rows-1)] + [10_000_000])

    def __check_rounding(self, test_id):
        for col_name in self.numeric_cols:
            vals = self.orig_df[col_name].fillna(-9595959484)
            vals = vals.astype(int)
            vals = vals.astype(str)
            vals = vals.replace('-9595959484', np.nan)

            num_zeros_arr = vals.str.replace('.0', '', regex=False).str.len() - \
                            vals.str.replace('.0', '', regex=False).str.strip('0').str.len()
            counts_series = num_zeros_arr.value_counts(normalize=True)
            last_normal = np.where(counts_series.sort_index().cumsum() > 0.995)[0][0]
            results_col = num_zeros_arr < (last_normal + 2)

            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                results_col,
                f'The column has consistently values with {last_normal} or less trailing zeros',
                f'with significantly more trailing zeros.',
            )

    def __generate_non_zero(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('non-zero rand', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('non-zero all ', [random.randint(0, 50)  for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('non-zero most', [random.randint(1, 100) for _ in range(self.num_synth_rows-1)] + [0])

    def __check_non_zero(self, test_id):
        for col_name in self.numeric_cols:
            test_series = np.array([x != 0 for x in self.orig_df[col_name]])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently contains non-zero values")

    def __generate_less_than_one(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('less_than_one rand', [random.random() * 2.0 for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('less_than_one all',  [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('less_than_one most', [random.random() for _ in range(self.num_synth_rows-1)] + [1.2])

    def __check_less_than_one(self, test_id):
        for col_name in self.numeric_cols:
            test_series = np.array([(abs(x) < 1.0) or is_missing(x) for x in self.orig_df[col_name]])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently contains values between -1.0, and 1.0 (exclusive)")

    def __generate_greater_than_one(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('greater_than_one rand', [random.random() * 2.0 for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('greater_than_one all',  [random.random() + 2.0 for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('greater_than_one most', [random.random() + 2.0 for _ in range(self.num_synth_rows-1)] + [0.6])

    def __check_greater_than_one(self, test_id):
        for col_name in self.numeric_cols:
            test_series = np.array([(x == 0) or (abs(x) >= 1.0) for x in self.orig_df[col_name].fillna(1)])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently contains absolute values greater than or equal to 1.0")

    #########################################################################
    # Data consistency checks for pairs of numeric columns
    #########################################################################

    def __generate_larger(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('larger rand', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('larger all ', [random.randint(200, 300) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('larger most', [random.randint(200, 300) for _ in range(self.num_synth_rows-1)] + [50])

    def __check_larger(self, test_id):
        num_pairs, col_pairs = self.__get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(col_pairs):,} combinations.")

            if self.column_medians[col_name_1] < (self.column_medians[col_name_2] * 0.95):
                continue

            sample_series = ((self.sample_df[col_name_1] - self.sample_df[col_name_2]) >= 0) | \
                            self.sample_df[col_name_1].isna() | \
                            self.sample_df[col_name_2].isna()
            if sample_series.tolist().count(False) > 1:
                continue

            test_series = ((self.orig_df[col_name_1] - self.orig_df[col_name_2]) >= 0) | \
                          self.orig_df[col_name_1].isna() | \
                          self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently larger than "{col_name_2}"')

    def __generate_much_larger(self):
        """
        Patterns without exceptions:
            'much larger all ' is consistently much larger than 'much larger rand', but this test is not in the
            shortlist of patterns to list.
        Patterns with exception:
            'much larger most' is consistently much larger than 'much larger rand', with one exception: row 99 has a
            similar value.
        """
        self.__add_synthetic_column(
            'much larger rand',
            [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'much larger all ',
            [random.randint(200_000, 300_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'much larger most',
            [random.randint(200_000, 300_000) for _ in range(self.num_synth_rows-1)] + [50])

    def __check_much_larger(self, test_id):

        # Test if col_name_1 is consistently 10x larger than col_name_2
        num_pairs, col_pairs = self.__get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(col_pairs):,} combinations.")

            if self.column_medians[col_name_1] < (self.column_medians[col_name_2] * 5.0):
                continue

            # Test on a sample
            sample_series = np.where(
                self.sample_df[col_name_2] != 0,
                (self.sample_df[col_name_1] / self.sample_df[col_name_2]) > 10.0,
                False
            )
            sample_series = sample_series | self.sample_df[col_name_1].isna() | self.sample_df[col_name_2].isna()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test on the full column
            test_series = np.where(
                self.orig_df[col_name_2] != 0,
                (self.orig_df[col_name_1] / self.orig_df[col_name_2]) > 10.0,
                False
            )
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()

            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently much larger than "{col_name_2}"',
                '(where values may still be larger, but not by the normal extent)')

    def __generate_similar_wrt_ratio(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('sim wrt ratio rand', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim wrt ratio all ', self.synth_df['sim wrt ratio rand'] * 1.01)
        self.__add_synthetic_column('sim wrt ratio most', self.synth_df['sim wrt ratio rand'] * 1.01)
        self.synth_df.loc[999, 'sim wrt ratio most'] = 100_000

    def __check_similar_wrt_ratio(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose and len(numeric_pairs_list) > 5000:
                if pair_idx % 1000 == 0:
                    print(f"  On pair {pair_idx} of {len(numeric_pairs_list)}")

            test_series_a = np.where(
                self.orig_df[col_name_2] != 0,
                abs((self.orig_df[col_name_1] / self.orig_df[col_name_2])),
                1.0
            )
            test_series = np.where((test_series_a > 0.5) & (test_series_a < 2.0), True, False)
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" and "{col_name_2}" have consistently similar values in terms of their ratio')

    def __generate_similar_wrt_difference(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        Patterns with exception:
        """
        self.__add_synthetic_column('sim wrt diff rand', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim wrt diff all ', self.synth_df['sim wrt diff rand'] - 2.5)
        self.__add_synthetic_column('sim wrt diff most', self.synth_df['sim wrt diff rand'] - 2.5)
        self.synth_df.loc[999, 'sim wrt diff most'] = 100_000

    def __check_similar_wrt_difference(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose and len(numeric_pairs_list) > 5000:
                if (pair_idx > 0) and (pair_idx % 1000) == 0:
                    print(f"  On pair {pair_idx} of {len(numeric_pairs_list)}")

            diff_medians = abs(self.column_medians[col_name_1] - self.column_medians[col_name_2])
            if diff_medians > min(self.column_medians[col_name_1], self.column_medians[col_name_2]):
                continue

            test_series = abs((self.orig_df[col_name_1] - self.orig_df[col_name_2])) < (min(self.orig_df[col_name_1].median(), self.orig_df[col_name_2].median()) / 10.0)
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                (f'"{col_name_1}" and "{col_name_2}" have consistently similar values in terms of their absolute '
                 f'difference'))

    def __generate_similar_to_inverse(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('sim to inv rand', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim to inv all ', 1 / self.synth_df['sim to inv rand'])
        self.__add_synthetic_column('sim to inv most', 1 / self.synth_df['sim to inv rand'])
        self.synth_df.loc[999, 'sim to inv most'] = 500

    def __check_similar_to_inverse(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            sample_series = [math.isclose(x, 1/y) if y != 0 else True
                             for x, y in zip(self.sample_df[col_name_1], self.sample_df[col_name_2])]
            if sample_series.count(False) > 1:
                continue

            test_series = np.array([math.isclose(x, 1/y) if y != 0 else True
                                    for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently the inverse of "{col_name_2}"')

    def __generate_similar_to_negative(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('sim to neg rand', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim to neg all ', -1 * self.synth_df['sim to neg rand'])
        self.__add_synthetic_column('sim to neg most', -1 * self.synth_df['sim to neg rand'])
        self.synth_df.loc[999, 'sim to neg most'] = 500

    def __check_similar_to_negative(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if len(numeric_pairs_list) > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            sample_series = [math.isclose(x, -y) if y != 0 else True
                             for x, y in zip(self.sample_df[col_name_1], self.sample_df[col_name_2])]
            if sample_series.count(False) > 1:
                continue

            test_series = np.array([math.isclose(x, -y)
                                    for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently the negative of "{col_name_2}"')

    def __generate_constant_sum(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('constant sum 1', [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('constant sum 2', 5000 - self.synth_df['constant sum 1'])
        self.__add_synthetic_column('constant sum 3', 5000 - self.synth_df['constant sum 1'])
        self.synth_df.at[999, 'constant sum 3'] = self.synth_df.at[999, 'constant sum 3'] * 2.0

    def __check_constant_sum(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if not self.check_columns_same_scale_2(col_name_1, col_name_2, order=10):
                continue
            sample_series = self.sample_df[col_name_1] + self.sample_df[col_name_2]
            nmad = np.nanmedian(np.absolute(sample_series - np.nanmedian(sample_series))) / np.nanmedian(sample_series) \
                if np.nanmedian(sample_series) != 0 \
                else 0.0
            if nmad > 0.01:
                continue

            test_series_a = self.orig_df[col_name_1] + self.orig_df[col_name_2]
            # Get the median absolute deviation of the differences, normalized by the median. Note, the scipy
            # implementation does not handle Null values.
            nmad = np.nanmedian(np.absolute(test_series_a - np.nanmedian(test_series_a))) / np.nanmedian(test_series_a) \
                if np.nanmedian(test_series_a) != 0 \
                else 0.0
            if nmad < 0.01:
                test_series = abs(test_series_a - np.nanmedian(test_series_a)) < \
                              abs(0.01 * np.nanmedian(test_series_a))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The sum of "{col_name_1}" and "{col_name_2}" is consistently close to '
                     f'{np.nanmedian(test_series_a)}')
                )

    def __generate_constant_diff(self):
        """
        Patterns without exceptions: the difference in 'constant diff 1' and 'constant diff 2' is consistently 1000
        Patterns with exception: the difference in 'constant diff 1' and 'constant diff 3' is consistently 1000 with
            the exception of row 999
        """
        self.__add_synthetic_column('constant diff 1', [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('constant diff 2', 1000 + self.synth_df['constant diff 1'])
        self.__add_synthetic_column('constant diff 3', 1000 + self.synth_df['constant diff 1'])
        self.synth_df.at[999, 'constant diff 3'] = self.synth_df.at[999, 'constant diff 3'] * 2.0

    def __check_constant_diff(self, test_id):
        # todo: in examples not flagged, don't include None/NaN -- do that always with exceptions for some tests i think
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if not self.check_columns_same_scale_2(col_name_1, col_name_2, order=100):
                continue

            # Test on a sample
            sample_series = self.sample_df[col_name_1] - self.sample_df[col_name_2]
            nmad = scipy.stats.median_abs_deviation(sample_series) / statistics.median(sample_series) \
                if statistics.median(sample_series) != 0 \
                else 0.0
            if nmad > 0.01:
                continue

            # Get the median absolute deviation of the differences, normalized by the median
            test_series_a = abs(self.orig_df[col_name_1] - self.orig_df[col_name_2])
            test_series_a = test_series_a.replace([np.inf, -np.inf, np.NaN], test_series_a.median())
            nmad = scipy.stats.median_abs_deviation(test_series_a) / statistics.median(test_series_a)\
                if statistics.median(test_series_a) != 0 \
                else 0.0
            if nmad < 0.01:
                test_series = abs(test_series_a - statistics.median(test_series_a)) < abs(0.01 * statistics.median(test_series_a))
                # todo: for all check_constant_* tests, check it wouldn't work as well to just use one column. Is the other just zero?
                # here, checking the same scale should work but doesn't seem to.

                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    f'The difference of "{col_name_1}" and "{col_name_2}" is consistently close to {statistics.median(test_series_a)}'
                )

    def __generate_constant_product(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('constant product 1',
                                    [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('constant product 2', 5000 / self.synth_df['constant product 1'])
        self.__add_synthetic_column('constant product 3', 5000 / self.synth_df['constant product 1'])
        self.synth_df.at[999, 'constant product 3'] = self.synth_df.at[999, 'constant product 3'] * 2.0

    def __check_constant_product(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            # Test on a sample
            sample_series = self.sample_df[col_name_1] * self.sample_df[col_name_2]
            nmad = scipy.stats.median_abs_deviation(sample_series) / statistics.median(sample_series) \
                if statistics.median(sample_series) != 0 \
                else 0.0
            if nmad > 0.01:
                continue

            test_series_a = abs(self.orig_df[col_name_1] * self.orig_df[col_name_2])
            # Get the median absolute deviation of the products, normalized by the median
            nmad = np.nanmedian(np.absolute(test_series_a - np.nanmedian(test_series_a))) / np.nanmedian(test_series_a) \
                if np.nanmedian(test_series_a) != 0 \
                else 0.0
            if nmad < 0.01:
                test_series = abs(test_series_a - np.nanmedian(test_series_a)) < abs(0.01 * np.nanmedian(test_series_a))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The product of "{col_name_1}" and "{col_name_2}" is consistently close to '
                     f'{np.nanmedian(test_series_a)}')
                )

    def __generate_constant_ratio(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('constant ratio 1', [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('constant ratio 2', 5000 * self.synth_df['constant ratio 1'])
        self.__add_synthetic_column('constant ratio 3', 5000 * self.synth_df['constant ratio 1'])
        self.synth_df.at[999, 'constant ratio 3'] = self.synth_df.at[999, 'constant ratio 3'] * 2.0

    def __check_constant_ratio(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        count_zeros_dict = {}
        for col_name in self.numeric_cols:
            count_zeros_dict[col_name] = self.num_rows - np.count_nonzero(self.orig_df[col_name])

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if (count_zeros_dict[col_name_1] > self.contamination_level) and (count_zeros_dict[col_name_2] > self.contamination_level):
                continue

            # If just col_name_2 has many zeros, swap the columns
            if count_zeros_dict[col_name_2] > self.contamination_level:
                temp = col_name_1
                col_name_1 = col_name_2
                col_name_2 = temp

            # Test first on a sample
            sample_series = self.sample_df[col_name_1] / self.sample_df[col_name_2]
            nmad = scipy.stats.median_abs_deviation(sample_series) / statistics.median(sample_series) \
                if statistics.median(sample_series) != 0 \
                else 0.0
            if nmad > 0.01:
                continue

            test_series_a = abs(self.orig_df[col_name_1] / self.orig_df[col_name_2])
            # Get the median absolute deviation of the ratios, normalized by the median
            nmad = np.nanmedian(np.absolute(test_series_a - np.nanmedian(test_series_a))) / np.nanmedian(test_series_a) \
                if np.nanmedian(test_series_a) != 0 \
                else 0.0
            if nmad < 0.01:
                test_series = abs(test_series_a - np.nanmedian(test_series_a)) < abs(0.01 * np.nanmedian(test_series_a))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The ratio of "{col_name_1}" and "{col_name_2}" is consistently close to '
                     f'{statistics.median(test_series_a)}')
                )

    def __generate_even_multiple(self):
        self.__add_synthetic_column('even multiple rand', [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('even multiple all ', random.randint(1, 1_000) * self.synth_df['even multiple rand'])
        self.__add_synthetic_column('even multiple most', self.synth_df['even multiple all '])
        self.synth_df.at[999, 'even multiple most'] = self.synth_df.at[999, 'even multiple most'] * 1.25

    def __check_even_multiple(self, test_id):
        """
        For each pair of numeric columns, A and B, this checks if A is consistently an even integer multiple of B.
        This skips rows where B is 0 or missing.
        """

        min_valid = self.num_rows / 2
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            num_valid = len(np.where(
                (self.orig_df[col_name_2] != 0) &
                (self.orig_df[col_name_2] != 1) &
                (self.orig_df[col_name_2] != -1))[0])
            if num_valid < min_valid:
                continue
            test_series = np.where(
                self.orig_df[col_name_2] != 0,
                self.orig_df[col_name_1] / self.orig_df[col_name_2],
                self.orig_df[col_name_1])
            test_series = np.array([float(x).is_integer() and x != 1 for x in test_series])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently an even integer multiple of "{col_name_2}"')

    def __generate_rare_combination(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        common_vals = []
        for i in range(8):
            for j in range(8):
                if i == 0 or i == 7 or j == 0 or j == 7:
                    common_vals.append([i, j])
        rare_vals = [[2, 2]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self.__add_synthetic_column('rare_combo all_a', data[:, 0])
        self.__add_synthetic_column('rare_combo all_b', data[:, 1])

    def __check_rare_combination(self, test_id):
        """
        This flags only pairs of values where the combination is unusual, but neither value by itself would be flagged.
        No rows are removed from the data during evaluation, only from the output once generated.
        If the data contains clusters, this will flag points outside of the clusters or on the fringes. If there is
        a single cluster, it will flag points on the fringes of that cluster. Data is often concentrated in the
        lower-left of the 2d space; in this case the test will flag any points far from the origin, which will typically
        be points with both values not quite extreme enough to be flagged in either single dimension.
        """
        def get_bins(col_name, num_bins):
            if self.orig_df[col_name].dropna().nunique() < num_bins:
                return None
            bins = [-np.inf]
            min_val = self.orig_df[col_name].min()
            max_val = self.orig_df[col_name].max()
            col_range = max_val - min_val
            bin_width = col_range / num_bins
            for i in range(num_bins-1):
                bins.append(min_val + (i * bin_width))
            bins.append(np.inf)
            bin_labels = [int(x) for x in range(len(bins)-1)]
            binned_values = pd.cut(self.orig_df[col_name], bins, labels=bin_labels)
            return binned_values

        column_bins = {}
        num_bins = 6
        for col_name in self.numeric_cols:
            column_bins[col_name] = get_bins(col_name, num_bins)

        cell_limit = math.ceil(self.contamination_level / 9)

        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 1_000 == 0:
                print(f"  Examining column set {pair_idx:,} of {len(numeric_pairs_list):,} combinations.")

            bins_1 = column_bins[col_name_1]
            bins_2 = column_bins[col_name_2]
            if (bins_1 is None) or (bins_2 is None):
                continue

            vc1 = bins_1.value_counts()
            vc2 = bins_2.value_counts()
            counts_col_1 = vc1.sort_index().values
            counts_col_2 = vc2.sort_index().values

            # There are 6x6=36 cells in the 2d space. However, this test does not flag values that are extreme in
            # either dimension, so does not flag the outer ring of cells, checking only the inner 4x4=16 cells.
            # Loop through the 16 internal cells and identify cells that both have low count and their 8 neighboring
            # cells have low count. We first get the counts of all 36 cells.
            cell_counts = np.zeros((num_bins, num_bins))
            test_series = [1] * self.num_rows
            for i in range(num_bins):
                subset_i = [(x, y) for x, y in zip(bins_1, bins_2) if x == i]
                for j in range(num_bins):
                    cell_counts[i][j] = len([1 for x, y in subset_i if y == j])
            flat_list = [item for sublist in cell_counts for item in sublist]

            # Check there are at least 9 cells with counts less than the contamination rate
            num_less_contamination = len([1 for x in flat_list if x < self.contamination_level])
            if num_less_contamination < 9:
                continue

            # Identify the internal cells with low counts.
            low_count_cells = []
            for i in range(1, num_bins-1):
                for j in range(2, num_bins-1):
                    if cell_counts[i][j] <= cell_limit:
                        if counts_col_1[i] > self.contamination_level and counts_col_2[j] > self.contamination_level:
                            low_count_cells.append((i, j))

            flagged_cells = []
            for i, j in low_count_cells:
                # These checks are ordered most to least likely to fail
                if cell_counts[i][j] == 0 or \
                   cell_counts[i][j-1] > cell_limit or \
                   cell_counts[i][j+1] > cell_limit or \
                   cell_counts[i-1][j] > cell_limit or \
                   cell_counts[i+1][j] > cell_limit or \
                   cell_counts[i-1][j-1] > cell_limit or \
                   cell_counts[i-1][j+1] > cell_limit or \
                   cell_counts[i+1][j-1] > cell_limit or \
                   cell_counts[i+1][j+1] > cell_limit:
                    continue
                flagged_cells.append((i, j))

            if len(flagged_cells) == 0:
                continue

            # Check if enough rows have been flagged that we do not consider this a pattern
            total_flagged = 0
            for i, j in flagged_cells:
                total_flagged += cell_counts[i][j]
            if total_flagged > self.contamination_level:
                continue

            # Loop through each flagged cell and flag all rows in those cells
            for i, j in flagged_cells:
                flagged_idxs = [x for x in bins_1.index if bins_1[x] == i and bins_2[x] == j]
                for f in flagged_idxs:
                    test_series[f] = False

            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                "One or more rare combinations of values were found",
                allow_patterns=False
            )

    def __generate_correlated(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        list_a = sorted([random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        list_b = sorted([random.randint(1, 2_000) for _ in range(self.num_synth_rows)])
        c = list(zip(list_a, list_b))
        random.shuffle(c)
        list_a, list_b = zip(*c)

        self.__add_synthetic_column('correlated rand_a', list_a)
        self.__add_synthetic_column('correlated rand_b', list_b)
        list_b = list(list_b)
        list_b[-1] = list_b[-1] * 10.0
        self.__add_synthetic_column('correlated most', list_b)

    def __check_correlated(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.orig_df[col_name_1].nunique(dropna=True) < self.contamination_level:
                continue
            if self.orig_df[col_name_2].nunique(dropna=True) < self.contamination_level:
                continue
            if np.array_equal(self.orig_df[col_name_1], self.orig_df[col_name_2], equal_nan=True):
                continue

            spearancorr = abs(self.orig_df[col_name_1].corr(self.orig_df[col_name_2], method='spearman'))
            if spearancorr >= 0.995:
                col_1_percentiles = self.orig_df[col_name_1].rank(pct=True)
                col_2_percentiles = self.orig_df[col_name_2].rank(pct=True)

                # Test for positive correlation
                test_series = np.array([abs(x-y) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_1}" is consistently similar, with regards to percentile, to "{col_name_2}"')

                # Test for negative correlation
                test_series = np.array([abs(x-(1.0 - y)) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    f'"{col_name_1}" AND "{col_name_2}"',
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_1}" is consistently inversely similar in percentile to "{col_name_2}"')

    def __generate_matched_zero(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('matched zero rand_a', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('matched zero rand_b', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('matched zero all', self.synth_df['matched zero rand_a'])
        self.__add_synthetic_column('matched zero most', self.synth_df['matched zero rand_a'])
        if self.synth_df.loc[999, 'matched zero most'] == 0:
            self.synth_df.loc[999, 'matched zero most'] = 1
        else:
            self.synth_df.loc[999, 'matched zero most'] = 0

    def __check_matched_zero(self, test_id):
        # todo: probably code this more like opposite_zeros below.
        for col_idx_1 in range(len(self.numeric_cols)-1):
            col_name_1 = self.numeric_cols[col_idx_1]
            col_1_zero_arr = [x == 0 for x in self.orig_df[col_name_1]]
            num_zero_1 = col_1_zero_arr.count(True)
            if num_zero_1 == 0:
                continue
            for col_idx_2 in range(col_idx_1 + 1, len(self.numeric_cols)):
                col_name_2 = self.numeric_cols[col_idx_2]
                col_2_zero_arr = [x == 0 for x in self.orig_df[col_name_2]]
                num_zero_2 = col_2_zero_arr.count(True)
                if num_zero_2 == 0:
                    continue

                # If the difference between the number of zero values is too large, there is not a pattern
                if abs(num_zero_1 - num_zero_2) > (self.contamination_level * 2.0):
                    continue

                test_series = np.array([x == y for x, y in zip(col_1_zero_arr, col_2_zero_arr)])
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The columns "{col_name_1}" (with {num_zero_1} zero values) and "{col_name_2}" (with '
                     f'{num_zero_2} zero values) consistently have zero values in the same rows')
                )

    def __generate_opposite_zero(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('opp_zero all_a',
                                    [random.randint(1, 500) if x % 2 == 0 else 0 for x in range(self.num_synth_rows)])
        self.__add_synthetic_column('opp_zero all_b',
                                    [random.randint(1000, 2000) if x % 2 == 1 else 0 for x in range(self.num_synth_rows)])
        self.__add_synthetic_column('opp_zero most', self.synth_df['opp_zero all_b'])
        self.synth_df.loc[998, 'opp_zero most'] = 600

    def __check_opposite_zero(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            zero_indicator_1 = self.orig_df[col_name_1] == 0
            num_zeros_1 = zero_indicator_1.tolist().count(True)
            zero_indicator_2 = self.orig_df[col_name_2] == 0
            num_zeros_2 = zero_indicator_2.tolist().count(True)
            if num_zeros_1 < (self.num_rows * 0.01) or \
                num_zeros_1 > (self.num_rows * 0.99) or \
                num_zeros_2 < (self.num_rows * 0.01) or \
                num_zeros_2 > (self.num_rows * 0.99):
                continue
            test_series = np.array([x != y for x, y in zip(zero_indicator_1, zero_indicator_2)])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                f"The columns consistently have zero values in the opposite rows"
            )

    def __generate_running_sum(self):
        """
        Patterns without exceptions: 'run_sum all ' is a running total of 'run sum rand'
        Patterns with exception: 'run_sum most' is a running total of 'run sum rand' with the exception of row 500.
            Note: this may be missed where Null values are added.
        """
        self.__add_synthetic_column('run_sum rand', [random.randint(1, 500) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('run_sum all ', self.synth_df['run_sum rand'].cumsum())
        self.__add_synthetic_column('run_sum most', self.synth_df['run_sum all '])
        self.synth_df.loc[500, 'run_sum most'] = 100

    def __check_running_sum(self, test_id):
        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            # This is more robust than checking the cumulative sum.
            test_series_a = self.orig_df[col_name_1].shift(1) + self.orig_df[col_name_2]
            test_series = self.orig_df[col_name_1] == test_series_a
            test_series.loc[0] = True  # The first row can not be tested for running totals.
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_1].shift(1).isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                f'Column "{col_name_1}" consistently contains a running sum of "{col_name_2}"',
                ''
            )

    def __generate_a_rounded_b(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        # todo: create more rand columns, so there's less overlap and smaller results

        self.__add_synthetic_column('a_rounded_b rand', [random.random() * 10_000 for _ in range(self.num_synth_rows)])

        # Test floor function
        self.__add_synthetic_column('a_rounded_b all_a', self.synth_df['a_rounded_b rand'].apply(np.floor))
        self.__add_synthetic_column('a_rounded_b most_a', self.synth_df['a_rounded_b all_a'])
        self.synth_df.loc[999, 'a_rounded_b most_a'] = 100.8

        # Test ceil function
        self.__add_synthetic_column('a_rounded_b all_b', self.synth_df['a_rounded_b rand'].apply(np.ceil))
        self.__add_synthetic_column('a_rounded_b most_b', self.synth_df['a_rounded_b all_b'])
        self.synth_df.loc[999, 'a_rounded_b most_b'] = 100.8

        # Test rounding to 1's
        self.__add_synthetic_column('a_rounded_b all_c', self.synth_df['a_rounded_b rand'].apply(np.round))
        self.__add_synthetic_column('a_rounded_b most_c', self.synth_df['a_rounded_b all_c'])
        self.synth_df.loc[999, 'a_rounded_b most_c'] = 100.8

        # Test rounding to 10's
        self.__add_synthetic_column('a_rounded_b all_d', (self.synth_df['a_rounded_b rand'] / 10).apply(round) * 10)
        self.__add_synthetic_column('a_rounded_b most_d', self.synth_df['a_rounded_b all_d'])
        self.synth_df.loc[999, 'a_rounded_b most_d'] = 100.8

        # Test rounding to 100's
        self.__add_synthetic_column('a_rounded_b all_e',(self.synth_df['a_rounded_b rand'] / 100).apply(round) * 100)
        self.__add_synthetic_column('a_rounded_b most_e', self.synth_df['a_rounded_b all_e'])
        self.synth_df.loc[999, 'a_rounded_b most_e'] = 100.8

        # Test rounding to 1000's
        self.__add_synthetic_column('a_rounded_b all_f', (self.synth_df['a_rounded_b rand'] / 1000).apply(round) * 1000)
        self.__add_synthetic_column('a_rounded_b most_f', self.synth_df['a_rounded_b all_f'])
        self.synth_df.loc[999, 'a_rounded_b most_f'] = 100.8

    def __check_a_rounded_b(self, test_id):

        # Cache the rounded, floor and ceiling values of each numeric column, as well as the number of digits after
        # the decimal point
        number_decimals_dict = {}

        sample_floor_dict = {}
        sample_ceil_dict = {}
        sample_round_dict = {}
        sample_round_10_dict = {}
        sample_round_100_dict = {}
        sample_round_1000_dict = {}

        floor_dict = {}
        ceil_dict = {}
        round_dict = {}
        round_10_dict = {}
        round_100_dict = {}
        round_1000_dict = {}

        for col_name in self.numeric_cols:
            number_decimals_dict[col_name] = self.orig_df[col_name].apply(get_num_digits)

            sample_floor_dict[col_name] = self.sample_df[col_name].apply(np.floor)
            sample_ceil_dict[col_name] = self.sample_df[col_name].apply(np.ceil)
            sample_round_dict[col_name] = self.sample_df[col_name].apply(np.round)
            sample_round_10_dict[col_name] = (self.sample_df[col_name] / 10).apply(np.round) * 10
            sample_round_100_dict[col_name] = (self.sample_df[col_name] / 100).apply(np.round) * 100
            sample_round_1000_dict[col_name] = (self.sample_df[col_name] / 1000).apply(np.round) * 1000

            floor_dict[col_name] = self.orig_df[col_name].apply(np.floor)
            ceil_dict[col_name] = self.orig_df[col_name].apply(np.ceil)
            round_dict[col_name] = self.orig_df[col_name].apply(np.round)
            round_10_dict[col_name] = (self.orig_df[col_name] / 10).apply(np.round) * 10
            round_100_dict[col_name] = (self.orig_df[col_name] / 100).apply(np.round) * 100
            round_1000_dict[col_name] = (self.orig_df[col_name] / 1000).apply(np.round) * 1000

        num_pairs, numeric_pairs_list = self.__get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            print((f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            # Exclude pairs of columns that are mostly identical
            num_same = (self.orig_df[col_name_1] == self.orig_df[col_name_2]).tolist().count(True)
            if num_same > (self.num_rows * 0.9):
                continue

            # Test floor function
            if number_decimals_dict[col_name_2].median() > 0:
                test_series = [is_missing(x) or is_missing(y) or x == math.floor(y)
                               for x, y in zip(self.sample_df[col_name_1], sample_floor_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [x == round(y) for x, y in zip(self.orig_df[col_name_1], floor_dict[col_name_2])]
                    test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                    if test_series.count(False) < self.contamination_level:
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name([col_name_2, col_name_1]),
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            f'Column "{col_name_1}" is consistently the same as the floor of "{col_name_2}"',)
                        continue

            # Test ceil function
            if number_decimals_dict[col_name_2].median() > 0:
                test_series = [x == math.floor(y) for x, y in zip(self.sample_df[col_name_1], sample_ceil_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [x == round(y) for x, y in zip(self.orig_df[col_name_1], ceil_dict[col_name_2])]
                    test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                    if test_series.count(False) < self.contamination_level:
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name([col_name_2, col_name_1]),
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            f'Column "{col_name_1}" is consistently the same as the ceiling of "{col_name_2}"',)
                        continue

            # Test rounding to a single digit
            test_series = [x == round(y) for x, y in zip(self.sample_df[col_name_1], sample_round_dict[col_name_2])]
            if test_series.count(False) <= 1:
                test_series = [x == round(y) for x, y in zip(self.orig_df[col_name_1], round_dict[col_name_2])]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                if test_series.count(False) < self.contamination_level:
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_2, col_name_1]),
                        [col_name_2, col_name_1],
                        np.array(test_series),
                        f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2}"',)
                    continue

            # Test rounding to 10's
            test_series = [x == round(y) for x, y in zip(self.sample_df[col_name_1], sample_round_10_dict[col_name_2])]
            if test_series.count(False) <= 1:
                test_series = [x == round(y) for x, y in zip(self.orig_df[col_name_1], round_10_dict[col_name_2])]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                if test_series.count(False) < self.contamination_level:
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_2, col_name_1]),
                        [col_name_2, col_name_1],
                        np.array(test_series),
                        f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2} to the 10s"',)
                    continue

            # Test rounding to 100's
            test_series = [x == round(y) for x, y in zip(self.sample_df[col_name_1], sample_round_100_dict[col_name_2])]
            if test_series.count(False) <= 1:
                test_series = [x == round(y) for x, y in zip(self.orig_df[col_name_1], round_100_dict[col_name_2])]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                if test_series.count(False) < self.contamination_level:
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_2, col_name_1]),
                        [col_name_2, col_name_1],
                        np.array(test_series),
                        f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2} to the 100s"',)
                    continue

            # Test rounding to 1000's
            test_series = [x == round(y) for x, y in zip(self.sample_df[col_name_1], sample_round_1000_dict[col_name_2])]
            if test_series.count(False) <= 1:
                test_series = [x == round(y) for x, y in zip(self.orig_df[col_name_1], round_1000_dict[col_name_2])]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                if test_series.count(False) < self.contamination_level:
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_2, col_name_1]),
                        [col_name_2, col_name_1],
                        np.array(test_series),
                        f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2} to the 1000s"',)
                    continue

    #########################################################################
    # Data consistency checks for pairs of columns, where one must be numeric
    #########################################################################

    def __generate_matched_zero_missing(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('matched zero miss rand_a', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('matched zero miss rand_b', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('matched zero miss all', self.synth_df['matched zero miss rand_a'])
        self.synth_df['matched zero miss all'] = self.synth_df['matched zero miss all'].replace(0, np.NaN)
        self.__add_synthetic_column('matched zero miss most', self.synth_df['matched zero miss rand_a'])
        self.synth_df['matched zero miss most'] = self.synth_df['matched zero miss most'].replace(0, np.NaN)
        if self.synth_df.loc[999, 'matched zero miss most'] == np.NaN:
            self.synth_df.loc[999, 'matched zero miss most'] = 1
        else:
            self.synth_df.loc[999, 'matched zero miss most'] = np.NaN

    def __check_matched_zero_missing(self, test_id):
        """
        This may occur, for example if the numeric column is "Number of Children" and the other column is
        "Age First Child". If there are zero children, we expect Null in the other column.
        """
        # todo: when list non-flagged, give examples all 4 combos
        for col_idx, col_name_1 in enumerate(self.numeric_cols):
            if self.verbose >= 2:
                if col_idx > 0 and col_idx % 50 == 0:
                    print(f"  Processing column: {col_idx} of {len(self.numeric_cols)} numeric columns)")
            col_1_zero_arr = [x == 0 for x in self.orig_df[col_name_1]]
            num_zero_1 = col_1_zero_arr.count(True)
            if num_zero_1 == 0:
                continue
            for col_name_2 in self.orig_df.columns:
                if col_name_1 == col_name_2:
                    continue
                col_2_missing_arr = [is_missing(x) for x in self.orig_df[col_name_2]]
                num_missing_2 = col_2_missing_arr.count(True)
                if num_missing_2 == 0:
                    continue

                # If the difference between the number of missing values is too large, there is not a pattern
                if abs(num_zero_1 - num_missing_2) > (self.contamination_level * 2.0):
                    continue

                test_series = np.array([x == y for x, y in zip(col_1_zero_arr, col_2_missing_arr)])
                self.__process_analysis_binary(
                    test_id,
                    f"{col_name_1} AND {col_name_2}",
                    [col_name_1, col_name_2],
                    test_series,
                    f'Where "{col_name_1}" is 0, "{col_name_2}" is consistently Null and not Null otherwise'
                )

    #########################################################################
    # Data consistency checks for sets of 3 numeric columns
    #########################################################################

    def __check_sum_exact(self, test_id):
        # todo: we can maybe combine this with checking for similar above, and adjust the threhold so there is a
        #   reasonable number of exceptions.
        pass

    def __generate_similar_to_diff(self):
        """
        Patterns without exceptions: 'similar_to_diff all' is the same as the difference in 'similar_to_diff rand_a' and
            'similar_to_diff rand_b'
        Patterns with exception: 'similar_to_diff most' is the same as the difference in 'similar_to_diff rand_a' and
            'similar_to_diff rand_b', with 1 exception
        """
        self.__add_synthetic_column('similar_to_diff rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('similar_to_diff rand_b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('similar_to_diff all',
                                    self.synth_df['similar_to_diff rand_a'] - self.synth_df['similar_to_diff rand_b'])
        self.__add_synthetic_column('similar_to_diff most', self.synth_df['similar_to_diff all'])
        self.synth_df.at[999, 'similar_to_diff most'] = 12.0

    def __check_similar_to_diff(self, test_id):
        num_combos = len(self.numeric_cols) * (len(self.numeric_cols) * (len(self.numeric_cols)-1)/2)
        if num_combos > self.max_combinations:
            print((f"  Skipping testing triples of numeric columns. There are {num_combos:,} triples. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for col_idx, col_name_3 in enumerate(self.numeric_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns.")
            _, column_pairs = self.__get_numeric_column_pairs_unique()
            for cols_idx, (col_name_1, col_name_2) in enumerate(column_pairs):
                if col_name_1 == col_name_3 or col_name_2 == col_name_3:
                    continue

                med_1 = abs(self.column_medians[col_name_1])
                med_2 = abs(self.column_medians[col_name_2])
                med_3 = abs(self.column_medians[col_name_3])

                # Test that subtracting the 1st and 2nd columns makes sense (they are on the same scale)
                if ((med_2 != 0) and ((med_1 / med_2) < 0.1)) or ((med_2 != 0) and ((med_1 / med_2) > 10.0)):
                    continue

                # Test that the columns may be related checking the medians of the 3 columns
                if med_3 > max(med_1, med_2):
                    continue
                if (med_3 < (abs(med_2 - med_1) * 0.5)) or (med_3 > (abs(med_2 - med_1) * 2.0)):
                    continue

                # Test the relationship on a small sample of the full data
                test_series_a = abs(self.sample_df[col_name_3] / abs(self.sample_df[col_name_1] - self.sample_df[col_name_2]))
                test_series_a = test_series_a.replace(np.NaN, 1.0)
                test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
                num_not_matching = test_series.tolist().count(False)
                if num_not_matching > 1:
                    continue

                # Test on the full data
                test_series_a = abs(self.orig_df[col_name_3] / abs(self.orig_df[col_name_1] - self.orig_df[col_name_2]))
                test_series_a = test_series_a.replace(np.NaN, 1.0)
                test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
                num_matching = test_series.tolist().count(True)
                if num_matching < (self.num_rows - self.contamination_level):
                    continue

                # The the match wouldn't be as close simply using col_name_1
                test_series_a = abs(1.0 - (self.orig_df[col_name_3] / abs(self.orig_df[col_name_1] - self.orig_df[col_name_2])))
                test_series_b = abs(1.0 - (self.orig_df[col_name_3] / abs(self.orig_df[col_name_1])))
                if test_series_a.median() < test_series_b.median():
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_1, col_name_2, col_name_3]),
                        [col_name_1, col_name_2, col_name_3],
                        test_series,
                        f"{col_name_3} is consistently similar to the difference of {col_name_1} and {col_name_2}")

    def __generate_diff_exact(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """

    def __check_diff_exact(self, test_id):
        pass

    def __generate_similar_to_product(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('similar to prod 1a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('similar to prod 1b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('similar to prod 2', self.synth_df['similar to prod 1a'] * self.synth_df['similar to prod 1b'])
        self.__add_synthetic_column('similar to prod 3', self.synth_df['similar to prod 2'])
        self.synth_df.at[999, 'similar to prod 3'] = 12.0

    def __check_similar_to_product(self, test_id):
        reported_dict = {}

        # todo: take the abs value of each numeric column here, create a new df, and just use that below.

        # Test if col_name_3 is approximately the product of col_name_1 and col_name_2
        num_triples, column_triples = self.__get_numeric_column_triples()
        if num_triples > self.max_combinations:
            print((f"  Skipping testing triples of numeric columns. There are {num_triples:,} triples. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 100_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(column_triples):,} combinations.")
            columns_tuple = tuple(sorted([col_name_1, col_name_2, col_name_3]))
            if columns_tuple in reported_dict:
                continue

            med_1 = abs(self.column_medians[col_name_1])
            med_2 = abs(self.column_medians[col_name_2])
            med_3 = abs(self.column_medians[col_name_3])

            # Test that the columns may be related checking the medians of the 3 columns
            if (med_3 < med_1) or (med_3 < med_2) or (med_3 < (med_1 * med_2 * 0.5)) or (med_3 > (med_1 * med_2 * 2.0)):
                continue

            # Test the relationship on a small sample of the full data
            test_series_a = self.sample_df[col_name_3] / (self.sample_df[col_name_1] * self.sample_df[col_name_2])
            test_series_a = test_series_a.replace(np.NaN, 1.0)
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching > 1:
                continue

            # Test on the full data
            test_series_a = self.orig_df[col_name_3] / (self.orig_df[col_name_1] * self.orig_df[col_name_2])
            test_series_a = test_series_a.replace(np.NaN, 1.0)
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            num_matching = test_series.tolist().count(True)
            if num_matching >= (self.num_rows - self.contamination_level):
                # todo: test that it's not that one column is 0 and col 3 is too
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2, col_name_3]),
                    [col_name_1, col_name_2, col_name_3],
                    test_series,
                    f'"{col_name_3}" is consistently similar to the product of "{col_name_1}" and "{col_name_2}"')
                reported_dict[columns_tuple] = True

    def __generate_product_exact(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """

    def __check_product_exact(self, test_id):
        pass

    def __generate_similar_to_ratio(self):
        """
        Patterns without exceptions: 'similar_to_ratio all' is consistently the ratio of rand_a / rand_b
        Patterns with exception: 'similar_to_ratio most' is consistently as well, but with one exception.
        """
        self.__add_synthetic_column('similar_to_ratio rand_a', [random.randint(1, 1000)
                                                                for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('similar_to_ratio rand_b', [random.randint(1, 1000)
                                                                for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('similar_to_ratio all', self.synth_df['similar_to_ratio rand_a'] /
                                    self.synth_df['similar_to_ratio rand_b'])
        self.__add_synthetic_column('similar_to_ratio most', self.synth_df['similar_to_ratio all'])
        self.synth_df.at[999, 'similar_to_ratio most'] = 12.0

    def __check_similar_to_ratio(self, test_id):
        # todo: take the abs value of each numeric column here, create a new df, and just use that below.

        # Test if col_name_3 is approximately the ratio of col_name_1 and col_name_2
        # There are 6 combinations. __get_numeric_column_triples() will return each. Each execution of the loop
        # we check if col_name_3 is similar to (col_name_1 / col_name_2)
        num_triples, column_triples = self.__get_numeric_column_triples()
        if num_triples > self.max_combinations:
            print((f"  Skipping testing triples of numeric columns. There are {num_triples:,} triples. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 100_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(column_triples):,} combinations.")

            # Test that the columns may be related just checking the medians of the 3 columns
            med_1 = self.column_medians[col_name_1]
            med_2 = self.column_medians[col_name_2]
            med_3 = self.column_medians[col_name_3]
            if med_2 != 0:
                ratio_meds_1_2 = abs(med_1 / med_2)
                if ratio_meds_1_2 != 0:
                    if ((med_3 / ratio_meds_1_2) < 0.5) or ((med_3 / ratio_meds_1_2) > 2.0):
                        continue

            # Test the relationship on a sample of the full data
            test_series_a = self.sample_df[col_name_3] / abs(self.sample_df[col_name_1] / self.sample_df[col_name_2])
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching > 1:
                continue

            # Test col1 / col2 on the full data
            test_series_a = self.orig_df[col_name_3] / abs(self.orig_df[col_name_1] / self.orig_df[col_name_2])
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_3].isna()
            num_matching = test_series.tolist().count(True)
            if num_matching >= (self.num_rows - self.contamination_level):
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2, col_name_3]),
                    [col_name_1, col_name_2, col_name_3],  # Put in order such that col_3 == col_1 / col_2
                    test_series,
                    f'"{col_name_3}" is consistently similar to the ratio of "{col_name_1}" and "{col_name_2}"')
                continue

    def __generate_ratio_exact(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        pass

    def __check_ratio_exact(self, test_id):
        pass

    def __generate_larger_than_sum(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('larger_sum rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('larger_sum rand_b', [random.randint(1000, 2000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('larger_sum all', self.synth_df['larger_sum rand_b'] + self.synth_df['larger_sum rand_a'] + random.randint(1, 10))
        self.__add_synthetic_column('larger_sum most', self.synth_df['larger_sum all'])
        self.synth_df.at[999, 'larger_sum most'] = self.synth_df.at[999, 'larger_sum most'] * 0.2

    def __check_larger_than_sum(self, test_id):
        """
        Where the 3 columns are A, B, and C, we try: A > (B+C)
        This is done for all 3 columns in the place of A, so there are 3 tests performed.
        """

        def test_larger(col_a, col_b, col_c):
            # Test the relationship on a small sample of the full data
            test_series = self.sample_df[col_c] > (self.sample_df[col_a] + self.sample_df[col_b])
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching > 1:
                return False

            test_series = self.orig_df[col_c] > (self.orig_df[col_a] + self.orig_df[col_b])
            test_series = test_series | self.orig_df[col_a].isna() | self.orig_df[col_b].isna() | self.orig_df[col_c].isna()
            if test_series.tolist().count(False) < self.contamination_level:
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_a, col_b, col_c]),
                    [col_a, col_b, col_c],
                    test_series,
                    (f'The values in "{col_c}" are consistently larger than the sum of "{col_a}", '
                     f' and "{col_b}"'))
                return True
            return False

        # todo: we should also check the spearman correlations match

        # Track which columns are mostly positive. We execute this test only on those columns.
        column_pos_dict = {}
        for col_name in self.numeric_cols:
            column_pos_dict[col_name] = \
                (((self.orig_df[col_name] >= 0) | self.orig_df[col_name].isna()).tolist().count(False) < self.contamination_level)

        num_triples, column_triples = self.__get_numeric_column_triples_unique()
        if num_triples > self.max_combinations:
            print((f"  Skipping testing triples of numeric columns. There are {num_triples:,} triples. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(column_triples):,} combinations.")

            if not self.check_columns_same_scale_3(col_name_1, col_name_2, col_name_3, order=5):
                continue

            # This test applies only to columns which are largely positive.
            if not column_pos_dict[col_name_1]:
                continue
            if not column_pos_dict[col_name_2]:
                continue
            if not column_pos_dict[col_name_3]:
                continue

            med_1 = self.column_medians[col_name_1]
            med_2 = self.column_medians[col_name_2]
            med_3 = self.column_medians[col_name_3]

            # Test if col_name_1 is larger than col_name_2 + col_name_3
            if (med_1 > med_2) and (med_1 > med_3) and (med_1 > (med_2 + med_3 * 0.5)):
                if test_larger(col_name_2, col_name_3, col_name_1):
                    continue

            # Test if col_name_2 is larger than col_name_1 + col_name_3
            if (med_2 > med_1) and (med_2 > med_3) and (med_2 > (med_1 + med_3 * 0.5)):
                if test_larger(col_name_1, col_name_3, col_name_2):
                    continue

            # Test if col_name_3 is larger than col_name_1 + col_name_2
            if (med_3 > med_1) and (med_3 > med_2) and (med_3 > (med_1 + med_2 * 0.5)):
                if test_larger(col_name_1, col_name_2, col_name_3):
                    continue

    def __generate_larger_than_abs_diff(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('larger_diff rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('larger_diff rand_b', [random.randint(1000, 2000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('larger_diff all', self.synth_df['larger_diff rand_b'] - self.synth_df['larger_diff rand_a'] + random.randint(1, 10))
        self.__add_synthetic_column('larger_diff most', self.synth_df['larger_diff all'])
        self.synth_df.at[999, 'larger_diff most'] = self.synth_df.at[999, 'larger_diff most'] * 0.2

    def __check_larger_than_abs_diff(self, test_id):
        """
        For example, with the blood-transfusion dataset on OpenML, there are columns for time since first donation and
        time since the last donation. The difference in these is the time between their first & last donations. There
        is also a column for number of donations. This cannot be larger than the difference between their first and
        last donation times. This test is only done where all 3 columns are on the same scale (none is more than
        double the other with respect to their medians).

        Where the 3 columns are A, B, and C, we try: C > abs(A-B)
        This is done for all 3 columns in the place of A, so there are 3 tests performed.
        """

        def test_larger(col_a, col_b, col_c):
            # Test the relationship col_c > abs(col_a - col_b) on a small sample of the full data
            test_series = self.sample_df[col_c] > abs(self.sample_df[col_a] - self.sample_df[col_b])
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching >= 1:
                return False

            # Test of the full dataset
            test_series = self.orig_df[col_c] > (self.orig_df[col_a] - self.orig_df[col_b])
            test_series = test_series | self.orig_df[col_a].isna() | self.orig_df[col_b].isna() | self.orig_df[col_c].isna()
            if test_series.tolist().count(False) < self.contamination_level:
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2, col_name_3]),
                    [col_a, col_b, col_c],  # Order such that the last depends on the others
                    test_series,
                    (f'The values in "{col_c}" are consistently larger than the difference between "{col_a}", '
                     f' and "{col_b}"'))
                return True
            return False

        # todo: we should also check the spearman correlations match. maybe. this catches something different.
        #   maybe 2 different tests.

        # Test if col_name_3 is consistently larger than the absolute difference in col_name_1 and col_name_2
        num_triples, column_triples = self.__get_numeric_column_triples_unique()
        if num_triples > self.max_combinations:
            print((f"  Skipping testing triples of numeric columns. There are {num_triples:,} triples. "
                   f"max_combinations is currently set to {self.max_combinations:,}"))
            return

        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(column_triples):,} combinations.")

            if not self.check_columns_same_scale_3(col_name_1, col_name_2, col_name_3, order=5):
                continue

            med_1 = abs(self.column_medians[col_name_1])
            med_2 = abs(self.column_medians[col_name_2])
            med_3 = abs(self.column_medians[col_name_3])

            # Test if col_name_1 is larger than abs(col_name_2 - col_name_3)
            if med_1 > (abs(med_2 - med_3) * 0.5):
                if test_larger(col_name_2, col_name_3, col_name_1):
                    continue

            # Test if col_name_2 is larger than abs(col_name_1 - col_name_3)
            if med_2 > (abs(med_1 - med_3) * 0.5):
                if test_larger(col_name_1, col_name_3, col_name_2):
                    continue

            # Test if col_name_3 is larger than abs(col_name_2 - col_name_3)
            if med_3 > (abs(med_1 - med_2) * 0.5):
                if test_larger(col_name_1, col_name_2, col_name_3):
                    continue

    ###############################################################################
    # Data consistency checks for numeric column in relation to all other numeric columns.
    ###############################################################################

    def __generate_sum_of_columns(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('sum of cols rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sum of cols rand_b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sum of cols rand_c', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sum of cols rand_d', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self.__add_synthetic_column('sum of cols all', self.synth_df[[
            'sum of cols rand_a',
            'sum of cols rand_b',
            'sum of cols rand_c']].sum(axis=1))
        self.__add_synthetic_column('sum of cols most', self.synth_df['sum of cols all'].copy())
        self.synth_df.at[999, 'sum of cols most'] = self.synth_df.at[999, 'sum of cols most'] * 5.0

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c plus a constant are always, and mostly
        # true
        self.__add_synthetic_column('sum of cols plus all', self.synth_df[[
            'sum of cols rand_a',
            'sum of cols rand_b',
            'sum of cols rand_c']].sum(axis=1))
        self.synth_df['sum of cols plus all'] += 67.3
        self.__add_synthetic_column('sum of cols plus most', self.synth_df['sum of cols plus all'].copy())
        self.synth_df.at[999, 'sum of cols plus most'] = self.synth_df.at[999, 'sum of cols plus most'] * 5.0

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c times a constant are always, and mostly
        # true
        self.__add_synthetic_column('sum of cols times all', self.synth_df[[
            'sum of cols rand_a',
            'sum of cols rand_b',
            'sum of cols rand_c']].sum(axis=1))
        self.synth_df['sum of cols times all'] *= 1.6
        self.__add_synthetic_column('sum of cols times most', self.synth_df['sum of cols times all'].copy())
        self.synth_df.at[999, 'sum of cols times most'] = self.synth_df.at[999, 'sum of cols times most'] * 5.0

    def __check_sum_of_columns(self, test_id):
        """
        Loop through each numeric column. For each, find a set of plausible other columns, and test each subset of
        these. Plausible columns have smaller, but not drastically smaller, medians to the current column.
        """

        # Track which columns are mostly positive. We execute this test only on those columns.
        column_pos_arr = []
        for col_name in self.numeric_cols:
            if ((self.orig_df[col_name] >= 0) | self.orig_df[col_name].isna()).tolist().count(False) < self.contamination_level:
                column_pos_arr.append(col_name)

        for col_idx, col_name in enumerate(column_pos_arr):
            if self.verbose >= 2:
                print(f"  Processing column: {col_name} ({col_idx + 1} of {len(column_pos_arr)} positive numeric columns)")

            # Identify the columns with a similar range
            printed_subset_size_msg = False
            similar_cols = []
            for c in self.numeric_cols:
                if c == col_name:
                    continue
                # We provide for a large range, which can be found with, for example, tax values and so on, which may
                # be only 1 to 15% of the total column
                if self.column_medians[col_name] > self.column_medians[c] > (self.column_medians[col_name] / 20.0):
                    # Ensure the columns are correlated with col_name. Any columns summing to col_name should be
                    corr = self.pearson_corr.loc[col_name, c]
                    if corr > 0.4:
                        similar_cols.append(c)
            if len(similar_cols) == 0:
                continue

            found_any = False

            # For any subsets whose sum is too small to match col_name, there is no use trying any smaller subsets.
            know_failed_subsets = {}

            for subset_size in range(len(similar_cols), 1, -1):
                if found_any:
                    break

                calc_size = math.comb(len(similar_cols), subset_size)
                skip_subsets = calc_size > self.max_combinations
                #print("subset_size:", subset_size, "calc_size:", calc_size, "skip: ", skip_subsets)
                if skip_subsets:
                    if self.verbose >= 2 and not printed_subset_size_msg:
                        print((f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. "
                               f"max_combinations is currently set to {self.max_combinations:,}"))
                        printed_subset_size_msg = True
                    continue

                subsets = list(combinations(similar_cols, subset_size))
                if self.verbose >= 2 and len(similar_cols) > 15:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
                for subset in subsets:

                    # Check if this subset is a subset of any subsets previously tried which were too small.
                    subset_matches = True
                    for kfs in know_failed_subsets:
                        if set(kfs).issuperset(set(subset)):
                            subset_matches = False
                            break
                    if not subset_matches:
                        continue

                    # Check if this set of columns summing to col_name is plausible, checking if the sum of medians is
                    # significantly smaller or larger. actually, no -- we ccheck ading/ multiply by constant below
                    sum_of_medians = 0
                    for c in subset:
                        sum_of_medians += self.column_medians[c]
                    if sum_of_medians > (self.column_medians[col_name] * 1.1):
                        continue

                    # todo: test all these on a sample first

                    subset = list(subset)
                    df = self.orig_df[subset]
                    col_sums = df.sum(axis=1)
                    diffs_series = self.orig_df[col_name] - col_sums
                    col_values = diffs_series == 0
                    col_values = self.check_results_for_null(col_values, col_name, subset)
                    if col_values.tolist().count(False) < self.contamination_level:
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name(subset + [col_name]),
                            subset + [col_name],
                            col_values,
                            f'The column "{col_name}" is consistently equal to the sum of the values in the columns {subset}',
                            ""
                        )
                        found_any = True
                        break
                    too_small_arr = self.orig_df[col_name] > col_sums
                    if too_small_arr.tolist().count(True) > self.contamination_level:
                        know_failed_subsets[tuple(subset)] = True

                    # Check if there is a constant difference between the column sums and the values in col_name. If so,
                    # column col_name is the sum of the columns plus a constant. We determine if the median value in
                    # the differences is this constant.
                    median_diff = diffs_series.median()
                    col_values = [math.isclose(x, median_diff) for x in diffs_series]
                    col_values = self.check_results_for_null(col_values, col_name, subset)
                    if col_values.tolist().count(False) < self.contamination_level:
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name(subset + [col_name]),
                            subset + [col_name],
                            np.array(col_values),
                            (f'The column "{col_name}" is consistently equal to the sum of the values in the columns '
                             f"{subset} plus {median_diff}"),
                            "",
                            display_info={"operation": "plus", "amount": median_diff}
                        )
                        found_any = True
                        break

                    # Check if there is a constant ratio between the column sums and the values in col_name. If so,
                    # column col_name is the sum of the columns times a constant. We determine if the median value in
                    # the differences is this constant.
                    ratios_series = self.orig_df[col_name] / col_sums
                    median_ratio = ratios_series.median()
                    col_values = [math.isclose(x, median_ratio) for x in ratios_series]
                    col_values = self.check_results_for_null(col_values, col_name, subset)
                    if col_values.tolist().count(False) < self.contamination_level:
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name(subset + [col_name]),
                            subset + [col_name],
                            np.array(col_values),
                            (f"The column {col_name} is consistently equal to the sum of the values in the columns "
                             f" {subset} times {median_ratio}"),
                            "",
                            display_info={"operation": "times", "amount": median_ratio}
                        )
                        found_any = True
                        break

    def __generate_min_of_columns(self):
        """
        Unlike sum_of_columns, there is not the concept of adding or multiplying a constant to the minimum of another
        set of columns.

        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('min_of_cols rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('min_of_cols rand_b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('min_of_cols rand_c', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('min_of_cols rand_d', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self.__add_synthetic_column('min_of_cols all', self.synth_df[[
            'min_of_cols rand_a',
            'min_of_cols rand_b',
            'min_of_cols rand_c']].min(axis=1))
        self.__add_synthetic_column('min_of_cols most', self.synth_df['min_of_cols all'].copy())
        self.synth_df.at[999, 'min_of_cols most'] = self.synth_df.at[999, 'min_of_cols most'] * 5.0

    def __check_min_of_columns(self, test_id):
        # Try all subsets where it's median is less than this column's, but more than 1/20 of it.
        # For each target column, try all subsets whose minimum values match this column.

        two_rows_np = self.sample_df[self.numeric_cols].sample(n=2).values

        for col_idx, col_name in enumerate(self.numeric_cols):
            if self.verbose >= 2:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")

            similar_cols = []
            similar_cols_idxs = []
            printed_subset_size_msg = False
            for c_idx, c in enumerate(self.numeric_cols):
                # We provide for a large range, which can be found with, for example, tax values and so on, which may
                # be only 1 to 15% of the total column
                if self.column_medians[col_name] < self.column_medians[c] < (self.column_medians[col_name] * 10.0):
                    # Ensure the columns are correlated with col_name. Any columns in the min to col_name should be.
                    corr = self.pearson_corr.loc[col_name, c]
                    if corr > 0.2:
                        similar_cols.append(c)
                        similar_cols_idxs.append(c_idx)
            if len(similar_cols) == 0:
                continue

            found_any = False
            for subset_size in range(len(similar_cols), 1, -1):
                if found_any:
                    break

                calc_size = math.comb(len(similar_cols), subset_size)
                skip_subsets = calc_size > self.max_combinations
                if skip_subsets:
                    if self.verbose >= 2 and not printed_subset_size_msg:
                        print((f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. max_combinations is "
                               f"currently set to {self.max_combinations:,}"))
                        printed_subset_size_msg = True
                    continue

                subsets = list(combinations(similar_cols_idxs, subset_size))
                if self.verbose >= 2 and len(similar_cols) > 15:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset in subsets:
                    subset = list(subset)

                    # Test on just 2 rows
                    test_np = two_rows_np[:, subset]
                    col_mins = test_np.min(axis=1)
                    test_series = np.where(two_rows_np[:, col_idx] == col_mins, True, False)
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on a subset of the rows
                    subset_names = np.array(self.numeric_cols)[subset].tolist()
                    df = self.sample_df[subset_names]
                    col_mins = df.min(axis=1)
                    test_series = np.where(self.sample_df[col_name].values == col_mins, True, False)
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on the full columns
                    df = self.orig_df[subset_names]
                    col_mins = df.min(axis=1)
                    diffs_series = self.orig_df[col_name] - col_mins
                    test_series = diffs_series == 0
                    test_series = self.check_results_for_null(test_series, col_name, subset_names)
                    if test_series.tolist().count(False) < self.contamination_level:
                        # Check if this is a pattern in a trivial way. Check each column in the subset is the min at
                        # least once.
                        subset_okay = True
                        for col in subset_names:
                            if [(x == y) for x, y, z in zip(self.orig_df[col], col_mins, test_series) if z].count(True) == 0:
                                subset_okay = False
                                break
                        if not subset_okay:
                            continue

                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name(subset_names + [col_name]),
                            subset_names + [col_name],
                            test_series,
                            (f'The column "{col_name}" is consistently equal to the minimum of the values in the '
                             f'columns {subset_names}'),
                            ""
                        )
                        found_any = True
                        break

    def __generate_max_of_columns(self):
        self.__add_synthetic_column('max_of_cols rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('max_of_cols rand_b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('max_of_cols rand_c', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('max_of_cols rand_d', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self.__add_synthetic_column('max_of_cols all', self.synth_df[[
            'max_of_cols rand_a',
            'max_of_cols rand_b',
            'max_of_cols rand_c']].max(axis=1))
        self.__add_synthetic_column('max_of_cols most', self.synth_df['max_of_cols all'].copy())
        self.synth_df.at[999, 'max_of_cols most'] = self.synth_df.at[999, 'max_of_cols most'] * 5.0

    def __check_max_of_columns(self, test_id):
        # Try all subsets where it's median is less than this column's, but more than 1/10 of it.
        # For each target column, try all subsets whose minimum values match this column.

        two_rows_np = self.sample_df[self.numeric_cols].sample(n=2).values

        for col_idx, col_name in enumerate(self.numeric_cols):
            if self.verbose >= 2:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")

            similar_cols = []
            similar_cols_idxs = []
            printed_subset_size_msg = False
            for c_idx, c in enumerate(self.numeric_cols):
                # We provide for a large range, which can be found with, for example, tax values and so on, which may
                # be only 1 to 15% of the total column
                if (self.column_medians[col_name] / 10.0) < self.column_medians[c] < self.column_medians[col_name]:
                    # Ensure the columns are correlated with col_name. Any columns in the min to col_name should be.
                    corr = self.pearson_corr.loc[col_name, c]
                    if corr > 0.2:
                        similar_cols.append(c)
                        similar_cols_idxs.append(c_idx)
            if len(similar_cols) == 0:
                continue

            found_any = False
            for subset_size in range(len(similar_cols), 1, -1):
                if found_any:
                    break

                calc_size = math.comb(len(similar_cols), subset_size)
                skip_subsets = calc_size > self.max_combinations
                if skip_subsets:
                    if self.verbose >= 2 and not printed_subset_size_msg:
                        print((f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. max_combinations is "
                               f"currently set to {self.max_combinations:,}"))
                        printed_subset_size_msg = True
                    continue

                subsets = list(combinations(similar_cols_idxs, subset_size))
                if self.verbose >= 2 and len(similar_cols) > 15:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset in subsets:
                    subset = list(subset)

                    # Test on just 2 rows
                    test_np = two_rows_np[:, subset]
                    col_maxs = test_np.max(axis=1)
                    test_series = np.where(two_rows_np[:, col_idx] == col_maxs, True, False)
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on a subset of the rows
                    subset_names = np.array(self.numeric_cols)[subset].tolist()
                    df = self.sample_df[subset_names]
                    col_maxs = df.max(axis=1)
                    test_series = np.where(self.sample_df[col_name].values == col_maxs, True, False)
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on the full columns
                    df = self.orig_df[subset_names]
                    col_maxs = df.max(axis=1)
                    diffs_series = self.orig_df[col_name] - col_maxs
                    test_series = diffs_series == 0
                    test_series = self.check_results_for_null(test_series, col_name, subset_names)
                    if test_series.tolist().count(False) < self.contamination_level:
                        # Check if this is a pattern in a trivial way. Check each column in the subset is the min at
                        # least once.
                        subset_okay = True
                        for col in subset_names:
                            if [(x == y) for x, y, z in zip(self.orig_df[col], col_maxs, test_series) if z].count(True) == 0:
                                subset_okay = False
                                break
                        if not subset_okay:
                            continue

                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name(subset_names + [col_name]),
                            subset_names + [col_name],
                            test_series,
                            (f'The column "{col_name}" is consistently equal to the maximum of the values in the '
                             f'columns {subset_names}'),
                            ""
                        )
                        found_any = True
                        break

    def __generate_mean_of_columns(self):
        """
        Unlike sum_of_columns, there is not the concept of adding or multiplying a constant to the minimum of another
        set of columns.

        Patterns without exceptions: None generated
        Patterns with exception: 'mean_of_cols most' and ''mean_of_cols all' are both similar to the means of a set of
            columns, including the other, as well as rand_a, rand_b, and rand_c
        """
        self.__add_synthetic_column('mean_of_cols rand_a', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('mean_of_cols rand_b', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('mean_of_cols rand_c', [random.random() for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('mean_of_cols rand_d', [random.random() for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self.__add_synthetic_column('mean_of_cols all', self.synth_df[[
            'mean_of_cols rand_a',
            'mean_of_cols rand_b',
            'mean_of_cols rand_c']].mean(axis=1))
        self.__add_synthetic_column('mean_of_cols most', self.synth_df['mean_of_cols all'].copy())
        self.synth_df.at[999, 'mean_of_cols most'] = self.synth_df.at[999, 'mean_of_cols most'] * 5.0

    def __check_mean_of_columns(self, test_id):
        """
        This test loops through each numeric column and determines if that column is the mean of some subset of the
        other numeric columns. Doing this, it is able to cache information from the first few columns checked,
        so subsequent columns tend to execute much faster.
        """

        # Track which columns are mostly positive. We execute this test only on those columns.
        pos_cols_list = []
        for col_name in self.numeric_cols:
            if ((self.orig_df[col_name] >= 0) | self.orig_df[col_name].isna()).tolist().count(False) < self.contamination_level:
                pos_cols_list.append(col_name)

        # We loop through the positive numeric columns and for each find the set of columns with similar values.
        # For each subset of these, we check if any is the mean of the others. We then do not need to check any of these
        # columns again, but continue through the positive numeric columns for numeric columns in other ranges.
        skip_col_sets = []

        for col_idx, col_name in enumerate(pos_cols_list):
            if self.verbose >= 2:
                print((f'  Processing column: "{col_name}" (and all numeric columns of similar ranges) ({col_idx + 1} '
                       f"of {len(pos_cols_list)} positive numeric columns)"))

            # Identify the columns with similar values, based on the medians
            printed_subset_size_msg = False
            similar_cols = [col_name]
            for c in pos_cols_list:
                # We provide for a large range, which can be found with, for example, tax values and so on, which may
                # be only 1 to 15% of the total column
                if c == col_name:
                    continue
                if (self.column_medians[col_name] / 5.0) < self.column_medians[c] < (self.column_medians[col_name] * 5.0):
                    # Ensure the columns are correlated with col_name. Any columns related to  col_name should be
                    # correlated.
                    corr = self.pearson_corr.loc[col_name, c]
                    if corr > 0.4:
                        similar_cols.append(c)
            if len(similar_cols) == 1:
                continue

            # Check if we've evaluated the same set, or a superset of this
            similar_cols_set = set(similar_cols)
            subset_matches = False
            for sct in skip_col_sets:
                if sct.issuperset(similar_cols_set):
                    subset_matches = True
                    break
            if subset_matches:
                continue
            skip_col_sets.append(similar_cols_set)

            if self.verbose >= 2:
                print(f"    There are {len(similar_cols_set)} columns with similar ranges of values.")
            found_any = False
            for subset_size in range(len(similar_cols), 2, -1):
                if found_any:
                    break

                calc_size = math.comb(len(similar_cols), subset_size)
                if calc_size > self.max_combinations:
                    if self.verbose >= 2 and not printed_subset_size_msg:
                        print((f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. max_combinations is "
                               f"currently set to {self.max_combinations:,}"))
                        printed_subset_size_msg = True
                    continue

                subsets = list(combinations(similar_cols, subset_size))
                if self.verbose >= 2 and len(similar_cols) > 15:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset in subsets:
                    # Check if this set of columns summing to col_name is plausible, checking if the mean of medians is
                    # significantly smaller or larger.
                    medians_arr = []
                    for c in subset:
                        medians_arr.append(self.column_medians[c])
                    mean_of_medians = statistics.mean(medians_arr)
                    if mean_of_medians < (self.column_medians[col_name] * 0.9):
                        continue
                    if mean_of_medians > (self.column_medians[col_name] * 1.1):
                        continue

                    subset = list(subset)

                    # Test on a sample of the rows in the columns. Using numpy works faster in this case.
                    cols_idxs = [self.orig_df.columns.tolist().index(x) for x in subset]
                    sample_np = self.sample_df.values[:, cols_idxs]
                    col_mean = sample_np.mean(axis=1)

                    # We loop through all the columns in the subset, to avoid duplicate work later
                    matching_column = []
                    for c in subset:
                        if np.array_equal(self.sample_df[c], col_mean):
                            matching_column = c
                            break
                    if not matching_column:
                        continue

                    # Test on the full columns
                    df = self.orig_df[subset].copy()
                    col_mean = df.mean(axis=1)
                    diffs_series = self.orig_df[matching_column] - col_mean
                    test_series = diffs_series == 0
                    test_series = self.check_results_for_null(test_series, col_name, subset)

                    # Get the set of columns to report, keeping the matching column the right-most
                    all_cols = list(set(subset + [col_name] + [matching_column]))
                    rest_of_cols = all_cols.copy()
                    rest_of_cols.remove(matching_column)
                    if test_series.tolist().count(False) < self.contamination_level:
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name(rest_of_cols + [matching_column]),
                            rest_of_cols + [matching_column],
                            test_series,
                            (f'The column "{matching_column}" is consistently equal to the mean of the values in the '
                             f'columns {rest_of_cols}'),
                            ""
                        )
                        found_any = True
                        break

    def generate_all_pos_or_all_neg(self):
        """
        Patterns without exceptions: None: 'all_pos_neg all' consistently has the same sign as 'all_pos_neg rand_a'.
            However, it is not reported as a pair of columns as it is part of a set of 3 features with near-perfect
            matching.
        Patterns with exception: 'all_pos_neg most' consistently has the same sign as 'all_pos_neg rand_a',
            and 'all_pos_neg all', with the exception of row 999.
        """
        self.__add_synthetic_column('all_pos_neg rand_a', [random.randint(-1000, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('all_pos_neg rand_b', [random.randint(-1000, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('all_pos_neg all', self.synth_df['all_pos_neg rand_a'] * random.randint(1, 100))
        self.__add_synthetic_column('all_pos_neg most', self.synth_df['all_pos_neg rand_a'] * random.randint(1, 100))
        self.synth_df.at[999, 'all_pos_neg most'] = self.synth_df.at[999, 'all_pos_neg most'] * -1.0

    def check_all_pos_or_all_neg(self, test_id):
        """
        We identify sets of numeric columns, not necessarily on the same scale, that are all both frequently
        positive and frequently negative. We then find the subsets of these columns that are consistently positive and
        negative together.
        """

        # Check all columns examined have at least 10% positive and 10% negative values.
        cols = []
        sample_pos_dict = {}
        sample_neg_dict = {}
        pos_dict = {}
        neg_dict = {}
        for col_idx, col_name in enumerate(self.numeric_cols):
            num_pos = len([x for x in self.orig_df[col_name] if x > 0])
            num_neg = len([x for x in self.orig_df[col_name] if x < 0])
            if num_pos > (self.num_rows * 0.1) and num_neg > (self.num_rows * 0.1):
                cols.append(col_name)

                # The sample data tends not to include Null values
                pos_arr = self.sample_df[col_name] > 0
                neg_arr = self.sample_df[col_name] < 0
                sample_pos_dict[col_name] = pos_arr
                sample_neg_dict[col_name] = neg_arr

                pos_arr = ((self.orig_df[col_name] > 0) | self.orig_df[col_name].isna())
                neg_arr = ((self.orig_df[col_name] < 0) | self.orig_df[col_name].isna())
                pos_dict[col_name] = pos_arr
                neg_dict[col_name] = neg_arr

        # Loop through the subsets, from biggest to smallest. Consider only subsets of at least 2 columns.
        # We may flag multiple subsets of the same size, but none that are smaller. This will allow some subsets to
        # be skipped, but is a useful heuristic to reduce execution time.
        num_cols = len(cols)
        found_any = False
        know_failed_subsets = {}
        for subset_size in range(num_cols, 1, -1):
            if found_any:
                break

            calc_size = math.comb(len(cols), subset_size)
            skip_subsets = calc_size > self.max_combinations
            if skip_subsets:
                if self.verbose >= 2:
                    print((f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. max_combinations is "
                           f"currently set to {self.max_combinations:,}"))
                continue

            subsets = list(combinations(cols, subset_size))
            if self.verbose >= 2:
                print(f"  Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset in subsets:

                # Check if this subset has already been determined to be impossible from a portion of a previously
                # checked subset
                subset_matches = True
                for kfs in know_failed_subsets:
                    if set(subset).issuperset(set(kfs)):
                        subset_matches = False
                        break
                if not subset_matches:
                    continue

                # Test on a sample of rows
                pos_matching_arr = [1] * len(self.sample_df)
                neg_matching_arr = [1] * len(self.sample_df)
                for c_idx, c in enumerate(subset[1:]):
                    pos_matching_arr = pos_matching_arr & (sample_pos_dict[subset[0]] == sample_pos_dict[c])
                    if pos_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                    neg_matching_arr = neg_matching_arr & (sample_neg_dict[subset[0]] == sample_neg_dict[c])
                    if neg_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                if not subset_matches:
                    know_failed_subsets[tuple(subset[:c_idx+2])] = True
                    continue

                # Test on the full columns
                pos_matching_arr = [1] * self.num_rows
                neg_matching_arr = [1] * self.num_rows
                subset_matches = True
                for c in subset[1:]:
                    pos_matching_arr = pos_matching_arr & (pos_dict[subset[0]] == pos_dict[c])
                    pos_matching_arr = self.check_results_for_null(pos_matching_arr, None, subset)
                    if pos_matching_arr.tolist().count(False) > self.contamination_level:
                        subset_matches = False
                        break
                    neg_matching_arr = neg_matching_arr & (neg_dict[subset[0]] == neg_dict[c])
                    neg_matching_arr = self.check_results_for_null(neg_matching_arr, None, subset)
                    if neg_matching_arr.tolist().count(False) > self.contamination_level:
                        subset_matches = False
                        break
                if not subset_matches:
                    continue
                found_any = True
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name(subset),
                    list(subset),
                    pos_matching_arr & neg_matching_arr,
                    f"The columns in {subset} are consistently positive and negative together",
                    ""
                )

    def generate_all_zero_or_all_non_zero(self):
        """
        Patterns without exceptions: None: 'all_pos_neg all' consistently has the same sign as 'all_pos_neg rand_a'.
            However, it is not reported as it is part of a set of 3 feature with near-perfect matching.
        Patterns with exception: 'all_pos_neg most' consistently has the same sign as 'all_pos_neg rand_a',
            and 'all_pos_neg all', with the exception of row 999.
        """
        self.__add_synthetic_column('all_zero_or_not rand_a', [random.randint(-2, 2) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('all_zero_or_not rand_b', [random.randint(-1000, 1000) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('all_zero_or_not all', self.synth_df['all_zero_or_not rand_a'] * random.randint(1, 100))
        self.__add_synthetic_column('all_zero_or_not most', self.synth_df['all_zero_or_not rand_a'] * random.randint(1, 100))
        if self.synth_df.at[999, 'all_zero_or_not most'] == 0:
            self.synth_df.at[999, 'all_zero_or_not most'] = 1
        else:
            self.synth_df.at[999, 'all_zero_or_not most'] = 0

    def check_all_zero_or_all_non_zero(self, test_id):
        """
        We identify sets of numeric columns, not necessarily on the same scale, that are all both frequently
        zero and frequently non-zer. We then find the subsets of these columns that are consistently zero and
        non-zero together.
        """
        # todo: when list non-flagged, give both cases
        # todo: in the pattern string, give the number of rows with & without zero for both columns

        # Check all columns examined have at least 10% positive and 10% negative values.
        cols = []
        sample_zero_dict = {}
        sample_non_zero_dict = {}
        zero_dict = {}
        non_zero_dict = {}
        for col_idx, col_name in enumerate(self.numeric_cols):
            num_zero = len([x for x in self.orig_df[col_name] if (x == 0) or is_missing(x)])
            num_non_zero = len([x for x in self.orig_df[col_name] if (x != 0) or is_missing(x)])
            if num_zero > (self.num_rows * 0.1) and num_non_zero > (self.num_rows * 0.1):
                cols.append(col_name)

                zero_dict[col_name] = self.orig_df[col_name] == 0
                non_zero_dict[col_name] = self.orig_df[col_name] != 0

                sample_zero_dict[col_name] = self.sample_df[col_name] == 0
                sample_non_zero_dict[col_name] = self.sample_df[col_name] != 0

        know_failed_subsets = {}

        # Loop through the subsets, from biggest to smallest. Consider only subsets of at least 2 columns.
        # We may flag multiple subsets of the same size, but none that are smaller. This will allow some subsets to
        # be skipped, but is a useful heuristic to reduce execution time.
        num_cols = len(cols)
        found_any = False
        for subset_size in range(num_cols, 1, -1):
            if found_any:
                break
            subsets = list(combinations(cols, subset_size))
            if self.verbose >= 2:
                print(f"  Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset in subsets:
                subset_matches = True

                # Check if this subset has already been determined to be impossible from a portion of a previously
                # checked subset
                for kfs in know_failed_subsets:
                    if set(subset).issuperset(set(kfs)):
                        subset_matches = False
                        break
                if not subset_matches:
                    continue

                sample_zero_matching_arr = [1] * len(self.sample_df)
                sample_non_zero_matching_arr = [1] * len(self.sample_df)
                zero_matching_arr = [1] * self.num_rows
                non_zero_matching_arr = [1] * self.num_rows

                # Test on a sample of rows
                for c_idx, c in enumerate(subset):
                    # We compare all columns to the first column in the set, so this must be identical
                    if c == subset[0]:
                        continue
                    sample_zero_matching_arr = sample_zero_matching_arr & (sample_zero_dict[subset[0]] == sample_zero_dict[c])
                    if sample_zero_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                    sample_non_zero_matching_arr = sample_non_zero_matching_arr & (sample_non_zero_dict[subset[0]] == sample_non_zero_dict[c])
                    if sample_non_zero_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                if not subset_matches:
                    know_failed_subsets[tuple(subset[:c_idx+1])] = True
                    continue

                # Test of the full set of rows
                for c in subset:
                    # We compare all columns to the first column in the set, so this must be identical
                    if c == subset[0]:
                        continue
                    zero_matching_arr = zero_matching_arr & (zero_dict[subset[0]] == zero_dict[c])
                    zero_matching_arr = self.check_results_for_null(zero_matching_arr, None, subset)
                    if zero_matching_arr.tolist().count(False) > self.contamination_level:
                        subset_matches = False
                        break
                    non_zero_matching_arr = non_zero_matching_arr & (non_zero_dict[subset[0]] == non_zero_dict[c])
                    non_zero_matching_arr = self.check_results_for_null(non_zero_matching_arr, None, subset)
                    if non_zero_matching_arr.tolist().count(False) > self.contamination_level:
                        subset_matches = False
                        break
                if not subset_matches:
                    continue
                found_any = True
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name(subset),
                    list(subset),
                    zero_matching_arr & non_zero_matching_arr,
                    "The columns are consistently all zero or all non-zero together.",
                    ""
                )

    def __generate_dt_regressor(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        def set_y(col1, col2):
            arr = []
            for i in range(self.num_synth_rows):
                if self.synth_df[col1][i] > 50:
                    if self.synth_df[col2][i] > 50:
                        arr.append(0.0 + random.randint(0, 5))
                    else:
                        arr.append(20.0 + random.randint(0, 5))
                else:
                    if self.synth_df[col2][i] > 50:
                        arr.append(40.0 + random.randint(0, 5))
                    else:
                        arr.append(60.0 + random.randint(0, 5))
            return arr

        self.synth_df['dt regr. 1a'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 1b'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 1c'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 1d'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 2'] = set_y('dt regr. 1a', 'dt regr. 1b')
        self.synth_df['dt regr. 3'] = set_y('dt regr. 1c', 'dt regr. 1d')
        self.synth_df.at[999, 'dt regr. 3'] = self.synth_df.at[999, 'dt regr. 3'] * 10.0

    def __check_dt_regressor(self, test_id):
        # We determine if it's possible to create a small, interpretable decision tree which is accurate and
        # substantially more accurate than a naive model. We check the train score on the decision tree, as
        # it is a constricted model so unlikely to overfit, and does not need to predict new instances, just
        # summarize well the existing data. We measure accuracy in terms of normalized root mean squared error.

        # Determine which columns to drop and which are categorical, and should be one-hot encoded
        drop_features = []
        categorical_features = []
        for col_name in self.orig_df.columns:
            if not pandas_types.is_numeric_dtype(self.orig_df[col_name]):
                if self.orig_df[col_name].nunique() > 5:
                    drop_features.append(col_name)
                else:
                    categorical_features.append(col_name)

        for col_idx, col_name in enumerate(self.numeric_cols):
            if (self.verbose >= 2) and (col_idx > 0) and (col_idx % 10) == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")

            # Skip columns that are largely zero
            if (self.num_rows - np.count_nonzero(self.orig_df[col_name])) > (self.num_rows / 2):
                continue

            regr = DecisionTreeRegressor(max_leaf_nodes=4)

            x_data = self.orig_df.drop(columns=[col_name])
            x_data = x_data.drop(columns=drop_features)
            x_data = pd.get_dummies(x_data, columns=categorical_features)
            for c in x_data.columns:
                x_data[c] = x_data[c].replace([np.inf, -np.inf, np.NaN], x_data[c].median())

            y = self.orig_df[col_name]
            y = y.replace([np.inf, -np.inf, np.NaN], y.median())

            # Remove the extreme values from y to make the predictor less fit to outliers
            upper_y = y.quantile(0.99)
            lower_y = y.quantile(0.01)
            train_y = y[(y > lower_y) & (y < upper_y)]
            train_x_data = x_data.loc[train_y.index]
            if len(train_x_data) < 50:
                continue

            regr.fit(train_x_data, train_y)
            y_pred = regr.predict(x_data)
            mae_dt = metrics.median_absolute_error(y, y_pred)
            mae_naive = metrics.median_absolute_error(y, [statistics.mean(y)] * len(y))
            # Normalize the MAE by dividing by the median (or mean if median is zero)
            if self.column_medians[col_name] != 0:
                norm_mae_dt = abs(mae_dt / self.column_medians[col_name])
                norm_mae_naive = abs(mae_naive / self.column_medians[col_name])
            else:
                norm_mae_dt = abs(mae_dt / self.orig_df[col_name].mean())
                norm_mae_naive = abs(mae_naive / self.orig_df[col_name].mean())

            if (norm_mae_dt < 0.1) and (norm_mae_dt < (norm_mae_naive / 8.0)):
                rules = tree.export_text(regr)
                # The export has the column names in the format 'feature_1' and so on. We replace these with the
                # actual column names
                cols = []
                for c_idx, c_name in enumerate(x_data.columns):
                    rule_col_name = f'feature_{c_idx}'
                    if rule_col_name in rules:
                        cols.append(c_name)
                        rules = rules.replace(rule_col_name, c_name)

                errors_arr = (y_pred - y)
                normalized_errors_arr = abs(errors_arr / self.column_medians[col_name])
                test_series = normalized_errors_arr < 0.1
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name(cols),
                    cols,
                    test_series,
                    f'All values in the column are consistently predictable based using a decision tree with the '
                    f'following rules: \n{rules}'
                )

    def __generate_lin_regressor(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('lin regr 1a', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('lin regr 1b', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('lin regr 1c', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('lin regr 1d', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('lin regr 1e', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('lin regr 1f', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'lin regr 2',
            (4.1 * self.synth_df['lin regr 1a']) + (2.1 * self.synth_df['lin regr 1b']) + (5.1 * self.synth_df['lin regr 1c']))
        self.__add_synthetic_column(
            'lin regr 3',
            (4.1 * self.synth_df['lin regr 1d']) + (2.1 * self.synth_df['lin regr 1e']) + (5.1 * self.synth_df['lin regr 1f']))
        self.synth_df.at[999, 'lin regr 3'] = self.synth_df.at[999, 'lin regr 3'] * 10.0

    def __check_lin_regressor(self, test_id):
        """
        We determine if it's possible to create a small, interpretable linear regression which is accurate and
        substantially more accurate than a naive model, which simply predicts the median. We check the in-sample score
        on the linear regression, as it is a constricted model so unlikely to over-fit, and does not need to predict
        new instances, just summarize well the existing data. We measure accuracy in terms of normalized root mean
        squared error. The linear regression uses only the numeric features in the data.
        """

        for col_idx, col_name in enumerate(self.numeric_cols):

            if (self.verbose >= 2) and (col_idx > 0) and (col_idx % 10) == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")

            # Skip columns that are largely zero
            if (self.num_rows - np.count_nonzero(self.orig_df[col_name])) > (self.num_rows / 2):
                continue

            regr = Lasso(alpha=100.0)

            x_data = self.orig_df.drop(columns=[col_name])
            x_data = x_data.drop(columns=self.binary_cols + self.date_cols + self.string_cols)
            for c in x_data.columns:
                x_data[c] = x_data[c].replace([np.inf, -np.inf, np.NaN], x_data[c].median())

            y = self.orig_df[col_name]
            y = y.replace([np.inf, -np.inf, np.NaN], y.median())

            # Remove the extreme values from y to make the predictor less fit to outliers
            upper_y = y.quantile(0.99)
            lower_y = y.quantile(0.01)
            train_y = y[(y > lower_y) & (y < upper_y)]
            train_x_data = x_data.loc[train_y.index]
            if len(train_x_data) < 50:
                continue

            # todo: min-max scale the features in order to be able to determine from the coefficients which are the
            #   most relevant. Then fit again using the original scales, but use the first coefficients to also
            #   indicate rough feature importances.

            # todo: use R2 instead of NRMSE.

            regr.fit(train_x_data, train_y)
            y_pred = regr.predict(x_data)
            mae_lr = metrics.median_absolute_error(y, y_pred)
            mae_naive = metrics.median_absolute_error(y, [statistics.median(y)] * len(y))

            # Normalize the MAE by dividing by the median (or mean if median is zero)
            if self.column_medians[col_name] != 0:
                norm_mae_lr = abs(mae_lr / self.column_medians[col_name])
                norm_mae_naive = abs(mae_naive / self.column_medians[col_name])
            else:
                norm_mae_lr = abs(mae_lr / self.orig_df[col_name].mean())
                norm_mae_naive = abs(mae_naive / self.orig_df[col_name].mean())

            if (norm_mae_lr < 0.1) and (norm_mae_lr < (norm_mae_naive / 10.0)):
                # Get a string representation of the linear regression formula
                regression_formula = f"{regr.intercept_:.3f} + "
                num_coefs = 0
                features_used = []
                for x_col_idx, x_col in enumerate(x_data.columns):
                    if regr.coef_[x_col_idx] >= 0.01:  # todo: this is too stringent if the columns are on different scales, but this does keep the formulas interpretable
                        regression_formula += f' {regr.coef_[x_col_idx]:.2f} * "{x_col}" + '
                        num_coefs += 1
                        features_used.append(x_col)
                features_used.append(col_name)
                regression_formula = regression_formula[:-2]  # remove the trailing + sign

                # Ignore cases where the column is a constant that can be predicted from a single intercept. There are
                # simpler tests for this case.
                if num_coefs == 0:
                    continue

                # Ignore cases where the column is a constant plus another column.
                if num_coefs == 1 and round(max(regr.coef_)) == 1.0:
                    continue

                # Determine if there are too many coefficients given the number of rows. If so, remove the least
                # predictive (not sure how to figure that out, maybe with 1d linear regressions -- can't use the
                # coefficients since not scaled, and when scale, the intercepts are weird) and see if we still have
                # a good model. todo: code this.

                # Determine if the formula is still accurate when using just the selected columns
                # todo: fill in

                errors_arr = (y_pred - y)
                normalized_errors_arr = abs(errors_arr / self.column_medians[col_name])
                test_series = normalized_errors_arr < 0.5
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name(features_used),
                    features_used,
                    test_series,
                    (f'The column "{col_name}" contains values that are consistently predictable based on a linear '
                     f'regression\nformula: {regression_formula}')
                )
                # todo: also save the results of the linear regresion and plot this against the actual values

    def __generate_small_vs_corr_cols(self):
        """
        Patterns without exceptions: None. This test does not flag patterns. There are 2 clusters of columns, but this
            is not flagged as a pattern.
        Patterns with exception: 'small_vs_corr_cols_2'
        """
        for i in range(5):
            self.__add_synthetic_column(f'small_vs_corr_cols_{i}', sorted([random.random() for _ in range(self.num_synth_rows)], reverse=False))
        self.synth_df.loc[999, 'small_vs_corr_cols_2'] = 0.0000001
        for i in range(3):
            self.__add_synthetic_column(f'small_vs_corr_cols_{i+5}', sorted([random.random() for _ in range(self.num_synth_rows)], reverse=True))
        for i in range(3):
            self.__add_synthetic_column(f'small_vs_corr_cols_{i+8}', [random.random() for _ in range(self.num_synth_rows)])

    def __get_column_clusters(self):
        def get_next_corr_pair(corr_matrix, correlated_sets_arr):
            for i in corr_matrix.index:
                for j in corr_matrix.index:
                    if i == j:
                        continue
                    if corr_matrix[i][j] > 0.9:
                        already_in_set = False
                        for csa in correlated_sets_arr:
                            if (i in csa) or (j in csa):
                                already_in_set = True
                        if already_in_set:
                            continue
                        return set((i, j))
            return None

        # todo: tighten up so they are all reasonably correlated with each other -- at least 0.9 with at least 1/2 of the other columns in the set
        def get_rest_set(corr_matrix, curr_set, correlated_sets_arr):
            for i in corr_matrix.index:
                for j in corr_matrix.index:
                    if i == j:
                        continue
                    if (corr_matrix[i][j] > 0.9) and ((i in curr_set) or (j in curr_set)):
                        for csa in correlated_sets_arr:
                            if (i in csa) or (j in csa):
                                continue
                        curr_set.add(i)
                        curr_set.add(j)
            return curr_set

        sub_df = self.orig_df[self.numeric_cols].sample(n=min(self.num_rows, 2000))
        corr_matrix = sub_df.corr(method='spearman')
        correlated_sets_arr=[]
        pair = get_next_corr_pair(corr_matrix, correlated_sets_arr)
        while pair:
            full_set = get_rest_set(corr_matrix, pair, correlated_sets_arr)
            correlated_sets_arr.append(full_set)
            pair = get_next_corr_pair(corr_matrix, correlated_sets_arr)
        return correlated_sets_arr

    def __check_small_vs_corr_cols(self, test_id):
        """
        This test is currently disabled. The idea is: for each column, we compare each value to the values in the
        same row for all correlated columns. This first finds the clusters of correlated columns. Probably instead
        it should just find, for each column, the set of columns that are reasonably correlated. Then it converts
        all values to the percentile relative to the column. It compares each percentile to the average percentile
        for that row. This works okay, but it may not give much more than comparing correlated columns pairwise, and
        is more difficult to explain and present.
        """
        correlated_sets_arr = self.__get_column_clusters()
        if len(correlated_sets_arr) == 0:
            return
        if self.verbose >= 2:
            print("  Identified the clusters of correlated columns: ", correlated_sets_arr)

        for cluster_idx, cluster in enumerate(correlated_sets_arr):
            if self.verbose >= 2:
                print(f"  Examining cluster {cluster_idx} of {len(correlated_sets_arr)} clusters of columns")
            df_rank = pd.DataFrame()
            for col_name in cluster:
                df_rank[col_name] = self.orig_df[col_name].rank(pct=True)
            df_rank['Avg Percentile'] = df_rank.mean(axis=1)

            for col_name in cluster:
                test_series = (df_rank['Avg Percentile'] - df_rank[col_name]) < 0.5
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    test_series,
                    (f'Column "{col_name}" contains values that are small compared to the values in similar columns: '
                     f'{cluster}'),
                    display_info={"cluster": cluster}
                )

    def __generate_large_vs_corr_cols(self):
        """
        Patterns without exceptions: None. This test does not flag patterns. There are 2 clusters of columns, but this
            is not flagged as a pattern.
        Patterns with exception: 'large_vs_corr_cols_2'
        """
        for i in range(3):
            self.__add_synthetic_column(f'large_vs_corr_cols_{i}', sorted([random.random() for _ in range(self.num_synth_rows)], reverse=False))
        for i in range(5):
            self.__add_synthetic_column(f'large_vs_corr_cols_{i+3}', sorted([random.random() for _ in range(self.num_synth_rows)], reverse=True))
        self.synth_df.loc[999, 'large_vs_corr_cols_5'] = 2.0
        for i in range(3):
            self.__add_synthetic_column(f'large_vs_corr_cols_{i+8}', [random.random() for _ in range(self.num_synth_rows)])

    def __check_large_vs_corr_cols(self, test_id):
        correlated_sets_arr = self.__get_column_clusters()
        if len(correlated_sets_arr) == 0:
            return

        if self.verbose >= 2:
            print("  Identified the clusters of correlated columns: ", correlated_sets_arr)

        for cluster_idx, cluster in enumerate(correlated_sets_arr):
            if self.verbose >= 2:
                print(f"  Examining cluster {cluster_idx} of {len(correlated_sets_arr)} clusters of columns")
            df_rank = pd.DataFrame()
            for col_name in cluster:
                df_rank[col_name] = self.orig_df[col_name].rank(pct=True)
            df_rank['Avg Percentile'] = df_rank.mean(axis=1)

            for col_name in cluster:
                test_series = ( df_rank[col_name] - df_rank['Avg Percentile']) < 0.5
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    test_series,
                    (f'Column "{col_name}" contains values that are large compared to the values in similar columns: '
                     f'{cluster}'),
                    display_info={"cluster": cluster}
                )

    #########################################################################
    # Data consistency checks for single date columns
    #########################################################################

    def __generate_early_dates(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('early date all ', pd.date_range(test_date, periods=self.num_synth_rows))
        self.__add_synthetic_column('early date most', pd.date_range(test_date, periods=self.num_synth_rows-1))
        self.synth_df.loc[999, 'early date most'] = datetime.datetime.strptime("01-7-2005", "%d-%m-%Y")

        # This function also tests that string columns that should be recognized as dates are properly converted to
        # dates. Test with the month in string format
        self.__add_synthetic_column('date_str_1',
                                    ['Jan 12 2021',
                                     'Jan. 13 2022 ',
                                     '01/01/2003'] +
                                    ['April 15, 2005'] * (self.num_synth_rows-3))
        # Test in YYYYMM format
        self.__add_synthetic_column('date_str_2',
                                    [202101,
                                     '2021/02',
                                     '2021-03'] +
                                    ['April 15, 2005'] * (self.num_synth_rows-3))

    def __check_early_dates(self, test_id):
        for col_name in self.date_cols:
            q1 = self.orig_df[col_name].quantile(0.25)
            q3 = self.orig_df[col_name].quantile(0.75)
            lower_limit = q1 - (self.iqr_limit * (q3 - q1))
            test_series = self.orig_df[col_name] > lower_limit
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f"The test flagged any values earlier than {lower_limit} as very early given the 25th quartile is "
                 f"{q1} and 75th is {q3}")
            )

    def __generate_late_dates(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('late date all ', pd.date_range(test_date, periods=self.num_synth_rows))
        self.__add_synthetic_column('late date most', pd.date_range(test_date, periods=self.num_synth_rows-1))
        self.synth_df.loc[999, 'late date most'] = datetime.datetime.strptime("01-7-2045", "%d-%m-%Y")

    def __check_late_dates(self, test_id):
        for col_name in self.date_cols:
            q1 = self.orig_df[col_name].quantile(0.25)
            q3 = self.orig_df[col_name].quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))  # Using a stricter threshold than the 2.2 normally used
            test_series = self.orig_df[col_name] < upper_limit
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f"The test flagged any values later than {upper_limit} as very late given the 25th quartile is "
                 f"{q1} and 75th is {q3}"),
            )

    # todo: also check for unusual dates (with distant neighbors)
    # difference in days between each entry:
    # pd.Series(sorted_test_series).diff().dt.days

    def __generate_unusual_dow(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('dow rand', pd.date_range(test_date, periods=self.num_synth_rows))
        self.__add_synthetic_column('dow all ', pd.date_range(test_date, periods=self.num_synth_rows, freq='W'))
        self.__add_synthetic_column('dow most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='W'))
        self.synth_df.loc[999, 'dow most'] = datetime.datetime.strptime("02-7-2022", "%d-%m-%Y")

    def __check_unusual_dow(self, test_id):
        for col_name in self.date_cols:
            # Check the dates span enough weeks to check the dow in a meaningful way
            min_date = self.orig_df[col_name].min()
            max_date = self.orig_df[col_name].max()
            if (max_date - min_date).days < 100:
                continue

            dow_list = pd.Series([x.day_of_week for x in self.orig_df[col_name]])
            counts_list = dow_list.value_counts()
            rare_dow = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.contamination_level]
            if len(rare_dow) == 0:
                continue
            test_series = np.array([x not in rare_dow for x in dow_list])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific days of the week',
                f"on the {rare_dow}th days of the week"
            )

    def __generate_unusual_dom(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('dom rand', pd.date_range(test_date, periods=self.num_synth_rows))
        self.__add_synthetic_column('dom all ', pd.date_range(test_date, periods=self.num_synth_rows, freq='M'))
        self.__add_synthetic_column('dom most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='M'))
        self.synth_df.loc[999, 'dom most'] = datetime.datetime.strptime("02-7-2022", "%d-%m-%Y")

    def __check_unusual_dom(self, test_id):
        for col_name in self.date_cols:
            # Check the dates span enough months to check the dom in a meaningful way
            min_date = self.orig_df[col_name].min()
            max_date = self.orig_df[col_name].max()
            if (max_date - min_date).days < 300:
                continue

            dom_list = pd.Series([x.day for x in self.orig_df[col_name]])
            counts_list = dom_list.value_counts()
            rare_dom = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.contamination_level]
            if len(rare_dom) == 0:
                continue
            test_series = np.array([x not in rare_dom for x in dom_list])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific days of the month',
                f"on the {rare_dom}th days of the month"
            )

    def __generate_unusual_month(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('month rand', pd.date_range(test_date, periods=self.num_synth_rows, freq='3D'))
        dates = pd.date_range(test_date, periods=self.num_synth_rows, freq='3D')
        dates = [x if x.month != 12 else x + relativedelta(months=1) for x in dates]
        self.__add_synthetic_column('month all ', dates)
        self.__add_synthetic_column('month most', dates)
        self.synth_df.loc[999, 'month most'] = datetime.datetime.strptime("02-12-2022", "%d-%m-%Y")

    def __check_unusual_month(self, test_id):
        for col_name in self.date_cols:
            # Check the dates span enough years to check the dom in a meaningful way
            min_date = self.orig_df[col_name].min()
            max_date = self.orig_df[col_name].max()
            if (max_date - min_date).days < (365 * 3):
                continue

            month_list = pd.Series([x.month for x in self.orig_df[col_name]])
            counts_list = month_list.value_counts()
            rare_months = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.contamination_level]
            if len(rare_months) == 0:
                continue
            test_series = np.array([x not in rare_months for x in month_list])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific months of the year',
                f"on the {rare_months}th months of the year"
            )

    def __generate_unusual_hour(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022 02:35:5", "%d-%m-%Y %H:%M:%S")
        self.__add_synthetic_column('hour rand', pd.date_range(test_date, periods=self.num_synth_rows, freq='s'))
        self.__add_synthetic_column('hour all ', pd.date_range(test_date, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('hour most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='D'))
        self.synth_df.loc[999, 'hour most'] = datetime.datetime.strptime("01-7-2022 04:35:5", "%d-%m-%Y %H:%M:%S")

    def __check_unusual_hour(self, test_id):
        for col_name in self.date_cols:
            # Check the times span enough days to check the dom in a meaningful way
            min_date = self.orig_df[col_name].min()
            max_date = self.orig_df[col_name].max()
            if (max_date - min_date).days < 5:
                continue

            # todo: don't check the hour is consistent if it's just consitently missing, which is common. check the
            #   minutes vary, or something.

            hour_list = pd.Series([x.hour for x in self.orig_df[col_name]])
            counts_list = hour_list.value_counts()
            rare_hours = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.contamination_level]
            if len(rare_hours) == 0:
                continue
            test_series = np.array([x not in rare_hours for x in hour_list])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific hours of the day',
                f"on the {rare_hours}th hours of the day"
            )

    def __generate_unusual_minutes(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022 02:35:5", "%d-%m-%Y %H:%M:%S")
        self.__add_synthetic_column('minutes rand', pd.date_range(test_date, periods=self.num_synth_rows, freq='s'))
        self.__add_synthetic_column('minutes all ', pd.date_range(test_date, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('minutes most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='D'))
        self.synth_df.loc[999, 'minutes most'] = datetime.datetime.strptime("01-7-2022 04:59:5", "%d-%m-%Y %H:%M:%S")

    def __check_unusual_minutes(self, test_id):
        for col_name in self.date_cols:
            # Check the times span enough days to check the dom in a meaningful way
            min_date = self.orig_df[col_name].min()
            max_date = self.orig_df[col_name].max()
            # Ensure there is at least 100 hours of range
            if ((max_date - min_date) / pd.Timedelta(hours=1)) < 100:
                continue

            # todo: don't check the hour is consistent if it's just consitently missing, which is common. check the
            #   minutes vary, or something.

            minutes_list = pd.Series([x.minute for x in self.orig_df[col_name]])
            counts_list = minutes_list.value_counts()
            rare_minutes = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.contamination_level]
            if len(rare_minutes) == 0:
                continue
            test_series = np.array([x not in rare_minutes for x in minutes_list])
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific minutes of the hour',
                f"on the {rare_minutes}th minutes of the hour"
            )

    #########################################################################
    # Data consistency checks pairs of date columns
    #########################################################################

    def __generate_constant_date_gap(self):
        """
        Patterns without exceptions: 'const_gap all_1' contains daily values starting July 1. ''const_gap all_2'
            contains daily values starting 1 week later, so is consistently 7 days after 'const_gap all_1'.
        Patterns with exception: 'const_gap most' is the same as 'const_gap all_2', so also has a consistent 7 day
            gap from 'const_gap all_1', but with one exception.
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        test_date2 = datetime.datetime.strptime("07-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('const_gap all_1', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('const_gap all_2 ', pd.date_range(test_date2, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('const_gap all_3 ', pd.date_range(test_date2, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('const_gap most', pd.date_range(test_date2, periods=self.num_synth_rows-1, freq='D'))
        self.synth_df.loc[999, 'const_gap most'] = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")

    def __check_constant_date_gap(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                gap_array = pd.Series([0 if (is_missing(x) or is_missing(y))
                                       else (x-y).days
                                       for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])])
                gap_array = gap_array | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_counts(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    gap_array,
                    f"Columns {col_name_1} and {col_name_2} are consistently",
                    "days apart"
                )

    def __generate_large_date_gap(self):
        """
        Patterns without exceptions: 'large_gap all_2' consistently has a small gap (1 to 10 days) after
            'large_gap all_1'
        Patterns with exception: 'large_gap most' consistently has a small gap (1 to 10 days) after
            'large_gap all_1, with 1 exception'
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('large_gap all_1', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('large_gap all_2', [self.synth_df.loc[x]['large_gap all_1'] + datetime.timedelta(days=random.randint(1, 10)) for x in self.synth_df.index] )
        self.__add_synthetic_column('large_gap most', [self.synth_df.loc[x]['large_gap all_1'] + datetime.timedelta(days=random.randint(1, 10)) for x in self.synth_df.index] )
        self.synth_df.loc[999, 'large_gap most'] = datetime.datetime.strptime("01-7-2028", "%d-%m-%Y")

    def __check_large_date_gap(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                if self.orig_df[col_name_1].median() > self.orig_df[col_name_2].median():
                    col_big = col_name_1
                    col_small = col_name_2
                else:
                    col_big = col_name_2
                    col_small = col_name_1
                gap_array = pd.Series([0 if (is_missing(x) or is_missing(y))
                                       else (x-y).days
                                       for x, y in zip(self.orig_df[col_big], self.orig_df[col_small])])
                gap_array_no_null = pd.Series([(x-y).days
                                               for x, y in zip(self.orig_df[col_big], self.orig_df[col_small])
                                               if not is_missing(x) and not is_missing(y)])
                q1 = gap_array_no_null.quantile(0.25)
                q3 = gap_array_no_null.quantile(0.75)
                iqr = q3 - q1
                threshold = q3 + (iqr * self.iqr_limit)
                test_series = gap_array < threshold
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    f'"{col_name_1}" AND "{col_name_2}"',
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The gap between "{col_name_1}" and "{col_name_2}" is larger than normal. We flag any gaps '
                     f'larger than {threshold} days, as the 25th percentile is {q1} days and the 75th {q3} days.'),
                    f"",
                    allow_patterns=False
                )

    def __generate_small_date_gap(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('small_gap all_1', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('small_gap all_2',
            [self.synth_df.loc[x]['small_gap all_1'] + datetime.timedelta(days=random.randint(15, 20)) for x in self.synth_df.index])
        self.__add_synthetic_column('small_gap most',
            [self.synth_df.loc[x]['small_gap all_1'] + datetime.timedelta(days=random.randint(15, 20)) for x in self.synth_df.index] )
        self.synth_df.loc[999, 'small_gap most'] = self.synth_df.loc[999, 'small_gap all_1']

    def __check_small_date_gap(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                if self.orig_df[col_name_1].median() > self.orig_df[col_name_2].median():
                    col_big = col_name_1
                    col_small = col_name_2
                else:
                    col_big = col_name_2
                    col_small = col_name_1
                gap_array = pd.Series([0 if (is_missing(x) or is_missing(y))
                                       else (x-y).days
                                       for x, y in zip(self.orig_df[col_big], self.orig_df[col_small])])
                gap_array_no_null = pd.Series([(x-y).days
                                               for x, y in zip(self.orig_df[col_big], self.orig_df[col_small])
                                               if not is_missing(x) and not is_missing(y)])
                q1 = gap_array_no_null.quantile(0.25)
                q3 = gap_array_no_null.quantile(0.75)
                iqr = q3 - q1
                threshold = q1 - (iqr * self.iqr_limit)
                test_series = gap_array > threshold
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    f'"{col_name_1}" AND "{col_name_2}"',
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The gap between "{col_name_1}" and "{col_name_2}" is smaller than normal. We flag any gaps '
                     f'smaller than {threshold} days, as the 25th percentile is {q1} days and the 75th {q3} days.'),
                    f"",
                    allow_patterns=False
                )

    def __generate_date_later(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column(
            'date_later all_1',
            pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column(
            'date_later all_2',
            [self.synth_df.loc[x]['date_later all_1'] + datetime.timedelta(days=random.randint(15, 20))
             for x in self.synth_df.index])
        self.__add_synthetic_column(
            'date_later most',
            [self.synth_df.loc[x]['date_later all_1'] + datetime.timedelta(days=random.randint(15, 20))
             for x in self.synth_df.index] )
        self.synth_df.loc[999, 'date_later most'] = test_date1

    def __check_date_later(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                if self.orig_df[col_name_1].median() > self.orig_df[col_name_2].median():
                    col_big = col_name_1
                    col_small = col_name_2
                else:
                    col_big = col_name_2
                    col_small = col_name_1
                gap_array = pd.Series([0 if (is_missing(x) or is_missing(y))
                                       else (x-y).days
                                       for x, y in zip(self.orig_df[col_big], self.orig_df[col_small])])
                test_series = gap_array > 0
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    f'"{col_name_1}" AND "{col_name_2}"',
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_big}" is consistently later than "{col_small}"',
                    f"",
                    allow_patterns=False
                )

    #########################################################################
    # Data consistency checks for two columns, where one is date and the other is numeric
    #########################################################################

    def __generate_large_given_date(self):
        """
        Patterns without exceptions: This method identifies exceptions, but not patterns.
        Patterns with exception: 'large_given_date all' is well correlated with the date and has no exceptions.
            'large_given_date most' has one value in the last date bin which is very large, even for that bin.
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('large_given_date rand', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('large_given_date all ', [x for x in range(self.num_synth_rows)])
        self.__add_synthetic_column('large_given_date most', [x for x in range(self.num_synth_rows - 1)] + [1_000_000])

    def __check_large_given_date(self, test_id):
        for date_col in self.date_cols:
            # Create 10 equal-width bins for the dates
            if self.orig_df[date_col].nunique() < 10:
                continue
            bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            bin_assignments = pd.cut(self.orig_df[date_col], bins=10, labels=bin_labels)
            if bin_assignments.isna().sum() > 0:
                bin_labels.append(-1)
                bin_assignments = pd.Series([-1 if is_missing(x) else x for x in bin_assignments])
            bin_row_idxs = {}
            for bin_id in bin_labels:
                bin_row_idxs[bin_id] = np.where(bin_assignments == bin_id)

            for num_col in self.numeric_cols:
                test_series = pd.Series()
                for bin_id in bin_labels:
                    sub_df = self.orig_df.loc[bin_row_idxs[bin_id]]
                    q1 = sub_df[num_col].quantile(0.25)
                    q3 = sub_df[num_col].quantile(0.75)
                    iqr = q3 - q1
                    threshold = q3 + (iqr * self.iqr_limit)
                    sub_test_series = pd.Series(sub_df[num_col] > threshold)
                    sub_test_series.index = list(bin_row_idxs[bin_id])
                    test_series = pd.concat([test_series, sub_test_series])
                test_series = test_series.sort_index()
                self.__process_analysis_binary(
                    test_id,
                    f'"{date_col}" AND "{num_col}"',
                    [date_col, num_col],
                    ~test_series,
                    f'"{num_col}" is unusually large given the date column: "{date_col}"',
                    f"",
                    allow_patterns=False
                )

    def __generate_small_given_date(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self.__add_synthetic_column('small_given_date rand', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self.__add_synthetic_column('small_given_date all ', [x for x in range(self.num_synth_rows)])
        self.__add_synthetic_column('small_given_date most', [x for x in range(self.num_synth_rows - 1)] + [2])

    def __check_small_given_date(self, test_id):
        for date_col in self.date_cols:
            # Create 10 equal-width bins for the dates
            if self.orig_df[date_col].nunique() < 10:
                continue
            bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            bin_assignments = pd.cut(self.orig_df[date_col], bins=10, labels=bin_labels)
            if bin_assignments.isna().sum() > 0:
                bin_labels.append(-1)
                bin_assignments = pd.Series([-1 if is_missing(x) else x for x in bin_assignments])
            bin_row_idxs = {}
            for bin_id in bin_labels:
                bin_row_idxs[bin_id] = np.where(bin_assignments == bin_id)

            for num_col in self.numeric_cols:
                test_series = pd.Series()
                for bin_id in bin_labels:
                    sub_df = self.orig_df.loc[bin_row_idxs[bin_id]]
                    q1 = sub_df[num_col].quantile(0.25)
                    q3 = sub_df[num_col].quantile(0.75)
                    iqr = q3 - q1
                    threshold = q1 - (iqr * self.iqr_limit)
                    sub_test_series = pd.Series(sub_df[num_col] < threshold)
                    sub_test_series.index = list(bin_row_idxs[bin_id])
                    test_series = pd.concat([test_series, sub_test_series])
                test_series = test_series.sort_index()
                self.__process_analysis_binary(
                    test_id,
                    f'"{date_col}" AND "{num_col}"',
                    [date_col, num_col],
                    ~test_series,
                    f'"{num_col}" is unusually small given the date column: "{date_col}"',
                    f"",
                    allow_patterns=False
                )

    #########################################################################
    # Data consistency checks for pairs of binary columns
    #########################################################################

    def __generate_binary_same(self):
        """
        Patterns without exceptions: 'bin sim all_1' and 'bin sim all_2' consistently have the same binary values
        Patterns with exception: 'bin sim most' compared to both 'bin sim all_1' and 'bin sim all_2', has the same
            values, but with 1 exception.

        We use values 7 & 9 to test where the binary values are not 0 and 1.
        """
        self.__add_synthetic_column('bin sim all_1', [7]*(self.num_synth_rows-500) + [9]*500)
        self.__add_synthetic_column('bin sim all_2', [7]*(self.num_synth_rows-500) + [9]*500)
        self.__add_synthetic_column('bin sim most', [7]*(self.num_synth_rows-501) + [9]*501)

    def __check_binary_same(self, test_id):
        # todo: all binary tests: do on all binary & numeric columns, but first convert the numeric to binary as
        #   being 0 or non-0.

        # If the dataset is large, test on a 2nd sample
        sample_1000_df = None
        if self.num_rows > 10_000:
            sample_1000_df = self.orig_df[self.binary_cols].sample(n=1000)

        pairs = self.__get_binary_column_pairs_unique(same_vocabulary=True)
        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 50 == 0:
                print(f"  Processing pair number {pair_idx:,} of {len(pairs):,} pairs of binary columns")

            # Test on a sample first
            test_series = np.array([x == y for x, y in zip(self.sample_df[col_name_1], self.sample_df[col_name_2])])
            test_series = test_series | self.sample_df[col_name_1].isna() | self.sample_df[col_name_2].isna()
            if test_series.tolist().count(False) > 1:
                continue

            # Test on a larger sample
            if sample_1000_df is not None:
                test_series = np.array([x == y for x, y in zip(sample_1000_df[col_name_1], sample_1000_df[col_name_2])])
                test_series = test_series | sample_1000_df[col_name_1].isna() | sample_1000_df[col_name_2].isna()
                if test_series.tolist().count(False) > 10:
                    continue

                # Consider imbalanced arrays. We also check the macro f1 score
                sub_df = sample_1000_df[[col_name_1, col_name_2]].dropna()
                f1score = f1_score(sub_df[col_name_1].astype(str), sub_df[col_name_2].astype(str), average='macro')
                if f1score < 0.75:
                    continue

            # Test on the full columns
            test_series = np.array([x == y for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            if test_series.tolist().count(False) > self.contamination_level:
                continue

            # Consider imbalanced arrays. We also check the macro f1 score
            sub_df = self.orig_df[[col_name_1, col_name_2]].dropna()
            f1score = f1_score(sub_df[col_name_1].astype(str), sub_df[col_name_2].astype(str), average='macro')
            if f1score < 0.9:
                continue

            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f"The columns consistently have the same value", "")

    def __generate_binary_opposite(self):
        """
        Patterns without exceptions: 'bin opp all_1' and 'bin opp all_2' consistently contain the opposite values.
        Patterns with exception: 'bin opp most' consistently has the opposite values as 'bin opp all_2', with one
            exception.
        """
        self.__add_synthetic_column('bin opp all_1', [0]*(self.num_synth_rows-500) + [1]*500)
        self.__add_synthetic_column('bin opp all_2', [1]*(self.num_synth_rows-500) + [0]*500)
        self.__add_synthetic_column('bin opp most',  [0]*(self.num_synth_rows-501) + [1]*501)

    def __check_binary_opposite(self, test_id):
        for col_name_1, col_name_2 in self.__get_binary_column_pairs_unique(same_vocabulary=True):
            if self.orig_df[col_name_1].isna().sum() > (self.num_rows / 2):
                continue
            if self.orig_df[col_name_2].isna().sum() > (self.num_rows / 2):
                continue

            test_series = np.array([x != y for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])])
            if test_series.tolist().count(False) > self.contamination_level:
                continue

            # Consider imbalanced arrays. We also check the macro f1 score
            v1, v2 = self.orig_df[col_name_2].dropna().unique()
            opp_arr = self.orig_df[col_name_2].map({v1: v2, v2: v1})
            opp_arr = opp_arr.fillna("NONE")
            # todo: flip col2 properly, using map, not ~ -- it may not be 0 & 1
            f1score = f1_score(self.orig_df[col_name_1].fillna("NONE").astype(str), opp_arr.astype(str), average='macro')
            if f1score < 0.9:
                continue

            self.__process_analysis_binary(
                test_id,
                f'"{col_name_1}" AND "{col_name_2}"',
                [col_name_1, col_name_2],
                test_series,
                f"The columns consistently have the opposite value", "")

    def __generate_binary_implies(self):
        """
        Patterns without exceptions: 'bin implies all_1' consistently implies 'bin implies all_2'. A value of 0 in
            'bin implies all_1' consistently implies a 1 in 'bin implies all_2'.
        Patterns with exception: 'bin implies all_1' consistently implies 'bin implies most', with 1 exception. A
            value of 1 in 'bin implies all_1' implies 'bin implies most' will have a value of 0, but does not in
            the last row.
        """
        self.__add_synthetic_column('bin implies all_1', [0]*(self.num_synth_rows-500) + [1]*500)
        self.__add_synthetic_column('bin implies all_2', [1]*(self.num_synth_rows-500) + [0]*500)
        self.__add_synthetic_column('bin implies most', [0]*(self.num_synth_rows-800) + [1]*799 + [0])

    def __check_binary_implies(self, test_id):
        """
        This essentially checks for rare combinations, and does not work well with columns with many Null values.
        """

        column_pairs = self.__get_binary_column_pairs_unique()
        for pair_idx, (col_name_1, col_name_2) in enumerate(column_pairs):
            if self.verbose >= 2:
                if pair_idx % 100 == 0:
                    print(f"  Examining pair of binary columns: {pair_idx} of {len(column_pairs)}")

            vals_1 = self.orig_df[col_name_1].dropna().unique()
            val_1a = vals_1[0]
            val_1b = vals_1[1]
            vals_2 = self.orig_df[col_name_2].dropna().unique()
            val_2a = vals_2[0]
            val_2b = vals_2[1]

            # Ensure there are at least 10% of the rows with both values in both columns.
            count_min = self.num_rows * 0.1

            mask_1a = (self.orig_df[col_name_1] == val_1a) | self.orig_df[col_name_1].isna()
            mask_1b = (self.orig_df[col_name_1] == val_1b) | self.orig_df[col_name_1].isna()
            mask_2a = (self.orig_df[col_name_2] == val_2a) | self.orig_df[col_name_2].isna()
            mask_2b = (self.orig_df[col_name_2] == val_2b) | self.orig_df[col_name_2].isna()

            count_1a = mask_1a.tolist().count(True)
            if count_1a < count_min:
                continue
            count_1b = mask_1b.tolist().count(True)
            if count_1b < count_min:
                continue
            count_2a = mask_2a.tolist().count(True)
            if count_2a < count_min:
                continue
            count_2b = mask_2b.tolist().count(True)
            if count_2b < count_min:
                continue

            mask_aa = mask_1a & mask_2a
            mask_ab = mask_1a & mask_2b
            mask_ba = mask_1b & mask_2a
            mask_bb = mask_1b & mask_2b

            count_aa = mask_aa.tolist().count(True)
            count_ab = mask_ab.tolist().count(True)
            count_ba = mask_ba.tolist().count(True)
            count_bb = mask_bb.tolist().count(True)

            test_series = None
            pattern_str = ""
            if count_aa < self.contamination_level:
                # The expected count is based on the fraction of value a in column 1 and value b in column two,
                # multiplied to the give the expected fraction of both, times the number of rows to give the expected
                # number of rows. This is then: (count_1a / self_num_rows) * (count_1a / self_num_rows) * self.num_rows
                # which is simplified below for efficiency
                expected_count = (count_1a / self.num_rows) * (count_2a)
                if count_aa < expected_count:
                    test_series = np.array(~mask_aa)
                    pattern_str = f'"{col_name_1}" value: {val_1a} consistently implies "{col_name_2}" value: {val_2b}'
            elif count_ab < self.contamination_level:
                expected_count = (count_1a / self.num_rows) * (count_2b)
                if count_ab < expected_count:
                    test_series = np.array(~mask_ab)
                    pattern_str = f'"{col_name_1}" value: {val_1a} consistently implies "{col_name_2}" value: {val_2a}'
            elif count_ba < self.contamination_level:
                expected_count = (count_1a / self.num_rows) * (count_2b)
                if count_ba < expected_count:
                    test_series = np.array(~mask_ba)
                    pattern_str = f'"{col_name_1}" value: {val_1b} consistently implies "{col_name_2}" value: {val_2b}'
            elif count_bb < self.contamination_level:
                expected_count = (count_1a / self.num_rows) * (count_2b)
                if count_bb < expected_count:
                    test_series = np.array(~mask_bb)
                    pattern_str = f'"{col_name_1}" value: {val_1b} consistently implies "{col_name_2}" value: {val_2a}'

            if test_series is not None:
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    pattern_str, "")

    #########################################################################
    # Data consistency checks for sets of binary columns
    #########################################################################
    def __generate_binary_and(self):
        """
        Patterns without exceptions: 'bin_and all' is consistently the AND of 'bin_and rand_1' and 'bin_and rand_2'
        Patterns with exception: 'bin_and most' is consistently the AND of 'bin_and rand_1' and 'bin_and rand_2' with
            1 exception.
        """
        self.__add_synthetic_column('bin_and rand_1', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('bin_and rand_2', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('bin_and all', self.synth_df['bin_and rand_1'] & self.synth_df['bin_and rand_2'])
        self.__add_synthetic_column('bin_and most', self.synth_df['bin_and rand_1'] & self.synth_df['bin_and rand_2'])
        self.synth_df.loc[999, 'bin_and most'] = (self.synth_df.loc[999, 'bin_and most'] + 1) % 2

    def __check_binary_and(self, test_id):
        def test_val(col_name, col_vals):
            def get_and(x):
                and_v = 1
                for v in x:
                    and_v = and_v & (is_missing(v) | v)
                return and_v

            # Determine which other columns may potentially be AND'd to match col_name.
            # Ensure all columns considered have the same two values and at least 10% of both values.
            # Also check they each have enough of the 2nd value (treated as equivalent to 1 in the binary sense)
            # to form an AND relationship with col_name.
            set_other_cols = []
            count_min = self.num_rows * 0.1
            val0, val1 = self.column_unique_vals[col_name]
            target_count_1 = self.orig_df[col_name].tolist().count(val1)

            for col_name_other in self.binary_cols:
                if col_name_other == col_name:
                    continue
                num_val0 = self.orig_df[col_name_other].tolist().count(val0)
                num_val1 = self.orig_df[col_name_other].tolist().count(val1)
                if num_val0 < count_min:
                    continue
                if num_val1 < count_min:
                    continue
                # If this column has less 1's then the target column, then no amount of ANDing with other columns
                # will allow it to match the target column
                if num_val1 < target_count_1:
                    continue
                num_ones_not_matching = len([1 for x, y in zip(self.orig_df[col_name], self.orig_df[col_name_other])
                                             if (((x == val1) and (y != val1)) and (not is_missing(x)) and (not is_missing(y)))])
                if num_ones_not_matching > self.contamination_level:
                    continue
                set_other_cols.append(col_name_other)

            # todo: skip subset sizes that would produce too many combinations. We can just calculate (n chose m)
            # todo: we can determine for each subset_size n, if there are no viable subsets, then there is no
            # use checking any smaller subsets. This occurs if, for each subset, the result of the AND operation
            # is such that smaller subsets could never do better. Need to code this to be actually faster than just
            # looping through.
            for subset_size in range(len(set_other_cols), 1, -1):
                subset_list = list(combinations(set_other_cols, subset_size))
                for subset_idx, subset in enumerate(subset_list):
                    if len(subset) > 1:
                        if self.verbose >= 2 and subset_idx > 0 and subset_idx % 1000 == 0:
                            print(f"    Examining subset of other binary columns: {subset_idx + 1} of "
                                  f"{len(subset_list)}")

                        # The AND can work based on either value if the column does not contain 0 and 1.
                        subset_df = self.orig_df[list(subset)].copy()
                        for c in subset_df.columns:
                            subset_df[c] = subset_df[c].fillna(-1).map({col_vals[0]: 0, col_vals[1]: 1}).fillna(-1).astype(int)

                        and_of_cols = subset_df.apply(get_and, axis=1)
                        test_series = np.array(self.orig_df[col_name] == and_of_cols)
                        test_series = self.check_results_for_null(test_series, col_name, subset)
                        num_matching = test_series.tolist().count(True)
                        if num_matching > (self.num_rows - self.contamination_level):
                            self.__process_analysis_binary(
                                test_id,
                                self.get_col_set_name(list(subset) + [col_name]),
                                list(subset) + [col_name],
                                test_series,
                                f'"{col_name}" is consistently the result of an AND operation over the columns {subset_list}',
                                ""
                            )
                            return

        for col_idx, col_name in enumerate(self.binary_cols):
            if self.verbose >= 2:
                print(f"  Processing column: {col_name} ({col_idx + 1} of {len(self.binary_cols)} binary columns)")
            col_vals = self.column_unique_vals[col_name]
            if self.orig_df[col_name].tolist().count(col_vals[0]) < (self.num_rows * 0.1) or \
                    self.orig_df[col_name].tolist().count(col_vals[1]) < (self.num_rows * 0.1):
                continue
            test_val(col_name, col_vals)

    def __generate_binary_or(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('bin_or rand_1', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('bin_or rand_2', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('bin_or all', self.synth_df['bin_or rand_1'] | self.synth_df['bin_or rand_2'])
        self.__add_synthetic_column('bin_or most', self.synth_df['bin_or rand_1'] | self.synth_df['bin_or rand_2'])
        self.synth_df.loc[999, 'bin_or most'] = (self.synth_df.loc[999, 'bin_or most'] + 1) % 2

    def __check_binary_or(self, test_id):
        def test_val(col_name, col_vals):
            def get_or(x):
                or_v = 0
                for v in x:
                    or_v = or_v | (is_missing(v) | v)
                return or_v

            # Determine which other columns may potentially be OR'd to match col_name.
            # Ensure all columns considered have at least 10% of both values. This also checks the 2 columns have the
            # same vocabulary.
            # Also check they each have few enough of the 2nd value (treated as equivalent to 1 in the binary sense)
            # to form an OR relationship with col_name.
            set_other_cols = []
            count_min = self.num_rows * 0.1
            val0, val1 = self.column_unique_vals[col_name]
            target_count_1 = self.orig_df[col_name].tolist().count(val1)
            for col_name_other in self.binary_cols:
                if col_name_other == col_name:
                    continue
                num_val0 = self.orig_df[col_name_other].tolist().count(val0)
                num_val1 = self.orig_df[col_name_other].tolist().count(val1)
                if num_val0 < count_min:
                    continue
                if num_val1 < count_min:
                    continue
                # If this column has more 1's then the target column, then no amount of ORing with other columns
                # will allow it to match the target column
                if num_val1 > target_count_1:
                    continue
                num_ones_not_matching = len([1 for x, y in zip(self.orig_df[col_name], self.orig_df[col_name_other])
                                             if ((x == val0) and (y == val1) and (not is_missing(x)) and (not is_missing(y)))])
                if num_ones_not_matching > self.contamination_level:
                    continue
                set_other_cols.append(col_name_other)

            for subset_size in range(len(set_other_cols), 1, -1):
                subset_list = get_subsets(set_other_cols)
                for subset_idx, subset in enumerate(subset_list):
                    if len(subset) > 1:
                        if self.verbose >= 2 and subset_idx > 0 and subset_idx % 1000 == 0:
                            print(f"    Examining subset of other binary columns: {subset_idx + 1} of "
                                  f"{len(subset_list)}")

                        # The OR can work based on either value if the column does not contain 0 and 1.
                        subset_df = self.orig_df[list(subset)].copy()
                        for c in list(subset_df.columns):
                            subset_df[c] = subset_df[c].fillna(-1).map({col_vals[0]: 0, col_vals[1]: 1}).fillna(-1).astype(int)

                        or_of_cols = subset_df[list(subset)].apply(get_or, axis=1)
                        test_series = np.array(self.orig_df[col_name] == or_of_cols)
                        test_series = self.check_results_for_null(test_series, col_name, subset)
                        num_matching = test_series.tolist().count(True)
                        if num_matching > (self.num_rows - self.contamination_level):
                            self.__process_analysis_binary(
                                test_id,
                                self.get_col_set_name(list(subset) + [col_name]),
                                list(subset) + [col_name],
                                test_series,
                                f'"{col_name}" is consistently the result of an OR operation over the columns {subset_list}',
                                ""
                            )
                            return True

        for col_idx, col_name in enumerate(self.binary_cols):
            if self.verbose >= 2:
                print(f"  Processing column: {col_name} ({col_idx + 1} of {len(self.binary_cols)} binary columns)")
            col_vals = self.column_unique_vals[col_name]
            if self.orig_df[col_name].tolist().count(col_vals[0]) < (self.num_rows * 0.1) or \
                    self.orig_df[col_name].tolist().count(col_vals[1]) < (self.num_rows * 0.1):
                continue
            test_val(col_name, col_vals)

    def __generate_binary_xor(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('bin_xor rand_1', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('bin_xor rand_2', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('bin_xor all', self.synth_df['bin_xor rand_1'] ^ self.synth_df['bin_xor rand_2'])
        self.__add_synthetic_column('bin_xor most', self.synth_df['bin_xor rand_1'] ^ self.synth_df['bin_xor rand_2'])
        self.synth_df.loc[999, 'bin_xor most'] = (self.synth_df.loc[999, 'bin_xor most'] + 1) % 2

    def __check_binary_xor(self, test_id):
        """
        This defines XOR for multiple inputs as "1" if exactly 1 input has value 1. However, as the binary columns
        may contain any pair of values, not necessarily "0" and "1", this test is repeated for both values.
        """
        reported_dict = {}
        for col_idx_1, col_name_1 in enumerate(self.binary_cols):

            if self.verbose >= 2:
                print(f"  Examining column {col_idx_1} of {len(self.binary_cols)} binary columns.")

            min_rows_per_value = self.num_rows * 0.1

            col_vals_1 = self.column_unique_vals[col_name_1]
            if self.orig_df[col_name_1].tolist().count(col_vals_1[0]) < min_rows_per_value or \
                    self.orig_df[col_name_1].tolist().count(col_vals_1[1]) < min_rows_per_value:
                continue

            for col_name_2, col_name_3 in self.__get_binary_column_pairs_unique():
                columns_tuple = tuple(sorted([col_name_1, col_name_2, col_name_3]))
                if columns_tuple in reported_dict:  # todo: maybe faster to remove
                    continue

                if col_name_1 == col_name_2 or col_name_1 == col_name_3:
                    continue

                col_vals_2 = self.column_unique_vals[col_name_2]
                if self.orig_df[col_name_2].tolist().count(col_vals_2[0]) < (self.num_rows * 0.1) or \
                        self.orig_df[col_name_2].tolist().count(col_vals_2[1]) < (self.num_rows * 0.1):
                    continue

                col_vals_3 = self.column_unique_vals[col_name_3]
                if self.orig_df[col_name_3].tolist().count(col_vals_3[0]) < (self.num_rows * 0.1) or \
                        self.orig_df[col_name_3].tolist().count(col_vals_1[1]) < (self.num_rows * 0.1):
                    continue

                # Test on sample first
                # todo: fill in

                # Test on the full columns
                vals_col_2 = self.orig_df[col_name_2].fillna(-1).map({col_vals_2[0]: 0, col_vals_2[1]: 1}).fillna(-1).astype(int)
                vals_col_3 = self.orig_df[col_name_3].fillna(-1).map({col_vals_3[0]: 0, col_vals_3[1]: 1}).fillna(-1).astype(int)
                test_series = np.array(self.orig_df[col_name_1] == (vals_col_2 ^ vals_col_3))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_3].isna()
                num_matching = test_series.tolist().count(True)
                if num_matching >= (self.num_rows - self.contamination_level):
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_2, col_name_3, col_name_1]),
                        [col_name_2, col_name_3, col_name_1],
                        test_series,
                        (f'"{col_name_1}" is consistently the result of an XOR operation with columns "{col_name_2}" '
                         f'and "{col_name_3}"'),
                        "")
                    reported_dict[columns_tuple] = True

    def __generate_binary_num_same(self):
        """
        Patterns without exceptions: None: the first 4 columns have consistently 2 "2" values and 2 "1" values. However,
            these will not be flagged, as they are part of a larger pattern with all 5 columns
        Patterns with exception: the full set of 5 columns have consistently 3 "1" values, other than the last row
        """
        # We have a set of 4 binary columns where there are always 2 set to 1
        vals = [[2, 2, 1, 1], [2, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1, 2], [1, 2, 2, 1]]
        data = np.array([vals[np.random.choice(range(len(vals)))] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('bin_num_same rand_a', data[:, 0])
        self.__add_synthetic_column('bin_num_same rand_b', data[:, 1])
        self.__add_synthetic_column('bin_num_same rand_c', data[:, 2])
        self.__add_synthetic_column('bin_num_same rand_d', data[:, 3])
        self.__add_synthetic_column('bin_num_same most', [1] * (self.num_synth_rows - 1) + [2])

    def __check_binary_num_same(self, test_id):
        """
        Identify sets of binary columns that have the same two values. For each such set, determine if there is a
        consistent number of these columns that has the same value. For example, if a set of 10 columns contains
        values 'y' and 'n', a pattern would exist if there are consistently, say, 7 columns with 'y' and 3 with 'n',
        though the specific columns with each value may vary from row to row.
        """

        # Each execution of the loop processes the sets of binary columns for a given pair of values (eg 0 and 1,
        # Y and N, etc). We first identify one binary column that has not been examined, and then find the other
        # binary columns with the same vocabulary.

        vals_processed = {}
        for col_idx, col_name in enumerate(self.binary_cols):
            col_vals_tuple = tuple(self.column_unique_vals[col_name])
            col_vals_set = set(col_vals_tuple)
            if col_vals_tuple in vals_processed:
                continue
            vals_processed[col_vals_tuple] = True
            if self.verbose >= 2:
                print(f"  Processing column: {col_name} ({col_idx+1} of {len(self.binary_cols)} binary columns) "
                      f"(and all other columns with values {col_vals_tuple}).")
            similar_cols = [col_name]
            for c_idx in range(col_idx + 1, len(self.binary_cols)):
                c_name = self.binary_cols[c_idx]
                c_vals = set(self.column_unique_vals[c_name])
                if len(col_vals_set.intersection(c_vals)) != 2:
                    continue
                similar_cols.append(c_name)
            if self.verbose >= 2:
                print(f"    There are {len(similar_cols)} binary columns with this pair of values")

            # Create a dataframe where all the values are mapped to 0 and 1. This allows us to perform sum() operations
            # by row on subsets of this (taking subsets of columns below).
            one_zero_df = self.orig_df[list(similar_cols)].copy()
            for c in one_zero_df.columns:
                one_zero_df[c] = one_zero_df[c].map({col_vals_tuple[0]: 0, col_vals_tuple[1]: 1}).fillna(0).astype(int)
            sample_one_zero_df = one_zero_df.sample(n=min(self.num_rows, 50))
            sample_one_zero_np = sample_one_zero_df.values

            found_any = False
            # Check subsets of at least size 3. With 2 columns, this is equivalent to the BINARY_OPPOSITE test
            for subset_size in range(len(similar_cols), 2, -1):
                if found_any:
                    break
                calc_size = math.comb(len(similar_cols), subset_size)
                skip_subsets = calc_size > self.max_combinations
                if skip_subsets:
                    if self.verbose >= 2:
                        print((f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. max_combinations is "
                               f"currently set to {self.max_combinations:,}"))
                    continue

                subsets = list(combinations(list(range(len(similar_cols))), subset_size))
                if self.verbose >= 2:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset in subsets:
                    # Test first on a sample of rows for the subset of columns
                    sample_subset_np = sample_one_zero_np[:, list(subset)]
                    sums = sample_subset_np.sum(axis=1)
                    count_most_common_sum = sums.tolist().count(statistics.mode(sums))
                    # Allow only 1 deviation for a small sample
                    if count_most_common_sum < (len(sample_one_zero_df) - 1):
                        continue

                    # If the sample seems to match, test on the full subset of columns
                    col_names = np.array(similar_cols)[list(subset)]
                    subset_df = one_zero_df[col_names].copy()
                    sums = subset_df.sum(axis=1)
                    vc = sums.value_counts(normalize=False).sort_values(ascending=False)
                    if vc.iloc[0] > (self.num_rows - self.contamination_level):
                        test_series = np.array([sums == vc.index[0]][0])
                        self.__process_analysis_counts(
                                test_id,
                                self.get_col_set_name(col_names),
                                list(col_names),
                                test_series,
                                (f"The set of columns: {col_names} consistently have exactly {vc.index[0]} "
                                 f"columns with value {col_vals_tuple[1]}."),
                                ""
                            )
                        found_any = True

    def __generate_binary_rare_combo(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        common_vals = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1],
        ]
        rare_vals = [1, 1, 1, 1, 1]

        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self.__add_synthetic_column('bin_rare_combo all_a', data[:, 0])
        self.__add_synthetic_column('bin_rare_combo all_b', data[:, 1])
        self.__add_synthetic_column('bin_rare_combo all_c', data[:, 2])
        self.__add_synthetic_column('bin_rare_combo all_d', data[:, 3])
        self.__add_synthetic_column('bin_rare_combo all_e', data[:, 4])

    def __check_binary_rare_combo(self, test_id):
        """
        This test considers all sets of binary columns of size 3 or more columns, where the columns have at least 10% of
        their values both values in that column. The columns do not need to have the same vocabulary.
        It identifies rare combinations of values. Rare is defined as occurring both 1) less than the contamination rate
        times; and 2) less than 1/20 as would occur given the number of columns in the set if the values were equally
        distributed.
        """
        cols = []
        min_count = self.num_rows / 10.0
        for col_name in self.binary_cols:
            val0, val1 = self.column_unique_vals[col_name]
            if self.orig_df[col_name].tolist().count(val0) > min_count and \
                    self.orig_df[col_name].tolist().count(val1) > min_count and \
                    self.orig_df[col_name].isna().sum() < min_count:
                cols.append(col_name)

        # We flag only combinations that occur less than would occur given a uniform joint distribution of values.
        max_subset_size = int(np.floor(np.log2(self.num_rows / 20)))
        cols_df = self.orig_df[cols].copy()
        for col_name in cols:
            cols_df[col_name] = cols_df[col_name].map({self.column_unique_vals[col_name][0]: 0,
                                                       self.column_unique_vals[col_name][1]: 1})
            cols_df[col_name] = cols_df[col_name].fillna(cols_df[col_name].mode()[0])

        cols_np = cols_df.values
        skipped_subsets = {}

        # Determine the upper limit on the number of distinct combinations that would allow any rows to be flagged.
        # We only flag a combination if its count is less than 1/50 what we would expect if all unique values
        # were equal count, given the number of unique values.
        max_combination_count = self.num_rows / 50

        for subset_size in range(3, min(len(cols), max_subset_size) + 1):
            subsets = list(combinations(range(len(cols)), subset_size))

            # Determine the upper limit for the count of a combination to be flagged. The count must be both below
            # the contamination rate and 1/10 of the expected count given a uniform joint distribution.
            flagged_limit = int(min(self.contamination_level, (self.num_rows / math.pow(2, subset_size)) / 10.0))

            if self.verbose >= 2:
                print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset in subsets:

                # Do not check subsets of columns if we have already flagged a subset of this subset. Note, this
                # may skip some rare combinations, but is an efficient method to reduce over-reporting. Also
                # do not check subsets of columns if a subset of this subset already has too many distinct combinations
                # to support flagging any combination.
                subset_matches = False
                for fs in skipped_subsets:
                    if set(subset).issuperset(set(fs)):
                        subset_matches = True
                        break
                if subset_matches:
                    continue

                subset_np = cols_np[:, list(subset)].astype(int)
                counts_info = np.unique(subset_np, axis=0, return_counts=True)

                if len(counts_info[1]) > max_combination_count:
                    skipped_subsets[tuple(subset)] = True

                sum_rare = sum([x for x in counts_info[1] if x < self.contamination_level])
                if 0 < sum_rare < self.contamination_level:
                    skipped_subsets[tuple(subset)] = True
                    test_series = [True] * self.num_rows

                    # Loop through each rare combination for this subset of columns
                    for count_idx, count_val in enumerate(counts_info[1]):
                        if count_val < flagged_limit and count_val < ((self.num_rows / len(counts_info[1])) / 50):
                            rare_vals = counts_info[0][count_idx]
                            sub_df = cols_df
                            for col_idx, col_name in enumerate(subset):
                                sub_df = sub_df[sub_df[cols[col_idx]] == rare_vals[col_idx]]
                            for i in sub_df.index:
                                test_series[i] = False
                            subset_names = [cols[x] for x in list(subset)]
                            self.__process_analysis_binary(
                                test_id,
                                self.get_col_set_name(subset_names),
                                subset_names,
                                test_series,
                                f'For columns {subset}, the combination of values is rare'
                            )

    #########################################################################
    # Data consistency checks for pairs of columns where one is binary and one is numeric
    #########################################################################

    def __generate_binary_matches_values(self):
        """
        Patterns without exceptions: 'bin_match_val all' is consistently 0 when 'bin_match_val rand_a' is under 600 and
            consistently 1 when it is over 600
        Patterns with exception: 'bin_match_val most' is the same, with 1 exception
        """
        self.__add_synthetic_column('bin_match_val rand_a', np.random.randint(1, 1000, self.num_synth_rows))
        self.__add_synthetic_column('bin_match_val all', self.synth_df['bin_match_val rand_a'] > 600)
        self.__add_synthetic_column('bin_match_val most', self.synth_df['bin_match_val all'])
        self.synth_df.loc[999, 'bin_match_val most'] = not self.synth_df.loc[999, 'bin_match_val most']

    def __check_binary_matches_values(self, test_id):
        """
        For each pair of columns, where one is binary and the other is numeric, determine if it's consistently the case
        that low values in the numeric column are associated with one value in the binary column, and high values with
        the other value in the binary column.
        """
        for bin_col in self.binary_cols:
            val0, val1 = self.column_unique_vals[bin_col]
            if self.orig_df[bin_col].tolist().count(val0) < (self.num_rows * 0.1):
                continue
            if self.orig_df[bin_col].tolist().count(val1) < (self.num_rows * 0.1):
                continue
            sub_df_0 = self.orig_df[self.orig_df[bin_col] == val0]
            sub_df_1 = self.orig_df[self.orig_df[bin_col] == val1]
            for num_col in self.numeric_cols:
                set_0_01_percentile = sub_df_0[num_col].quantile(0.01)
                set_0_99_percentile = sub_df_0[num_col].quantile(0.99)
                set_1_01_percentile = sub_df_1[num_col].quantile(0.01)
                set_1_99_percentile = sub_df_1[num_col].quantile(0.99)

                # If the numeric values tend to be larger for value 0
                if set_0_01_percentile > set_1_99_percentile:
                    # Test if the binary column is consistently vol_0 for the larger values in the numeric column
                    threshold = statistics.mean(set_0_01_percentile, set_1_99_percentile)
                    test_series = [True if (x == val0 and y > threshold) or (x == val1 and y <= threshold) else False
                                   for x, y in zip(self.orig_df[bin_col], self.orig_df[num_col])]
                    test_series = test_series | self.orig_df[bin_col].isna() | self.orig_df[num_col].isna()
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([num_col, bin_col]),
                        [num_col, bin_col],
                        test_series,
                        (f'Column "{bin_col}" is consistently "{val1}" when Column "{num_col}" contains values under '
                         f'{threshold} and {val0} when values are over {threshold}',)
                    )

                # If the numeric values tend to be larger for value 0
                if set_1_01_percentile > set_0_99_percentile:
                    # Test if the binary column is consistently vol_1 for the larger values in the numeric column
                    threshold = statistics.mean([set_1_01_percentile, set_0_99_percentile])
                    test_series = [True if (x == val1 and y > threshold) or (x == val0 and y <= threshold) else False
                                   for x, y in zip(self.orig_df[bin_col], self.orig_df[num_col])]
                    test_series = test_series | self.orig_df[bin_col].isna() | self.orig_df[num_col].isna()
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([bin_col, num_col]),
                        [bin_col, num_col],
                        test_series,
                        (f'Column "{bin_col}" is consistently "{val0}" when Column "{num_col}" contains values under '
                         f'{threshold} and {val1} when values are over {threshold}',)
                    )

    #########################################################################
    # Data consistency checks for sets of three columns, where one must be binary
    #########################################################################

    def __generate_binary_two_others_match(self):
        """
        Patterns without exceptions: 'bin_match_others all' is constistently true iff 'bin_match_others rand_a' ==
            'bin_match_others rand_b'
        Patterns with exception: similar for 'bin_match_others most', with 1 exception
        """
        self.__add_synthetic_column('bin_match_others rand_a', np.random.randint(1, 10, self.num_synth_rows))
        self.__add_synthetic_column('bin_match_others rand_b', np.random.randint(1, 5, self.num_synth_rows))
        self.__add_synthetic_column('bin_match_others all', self.synth_df['bin_match_others rand_a'] == self.synth_df['bin_match_others rand_b'])
        self.__add_synthetic_column('bin_match_others most', self.synth_df['bin_match_others all'])
        self.synth_df.loc[999, 'bin_match_others most'] = not self.synth_df.loc[999, 'bin_match_others most']

    def __check_binary_two_others_match(self, test_id):

        def test_set(bin_col, col_name_2, col_name_3):

            # When checking pairs of binary columns, one may be the same as the bin_col
            if bin_col == col_name_2 or bin_col == col_name_3:
                return

            # Test first on a sample of the columns
            if sample_sub_df_0[col_name_2].dtype.name == 'category' or \
                    sample_sub_df_0[col_name_3].dtype.name == 'category':
                test_series_0 = sample_sub_df_0[col_name_2].astype(str) == sample_sub_df_0[col_name_3].astype(str)
                test_series_1 = sample_sub_df_1[col_name_2].astype(str) == sample_sub_df_1[col_name_3].astype(str)
            else:
                test_series_0 = sample_sub_df_0[col_name_2] == sample_sub_df_0[col_name_3]
                test_series_1 = sample_sub_df_1[col_name_2] == sample_sub_df_1[col_name_3]

            # Test if the binary column has val_0 iff the other 2 columns are equal
            sample_okay_val_0 = False
            if (test_series_0.tolist().count(False) < 1) and (test_series_1.tolist().count(True) < 1):
                sample_okay_val_0 = True

            # Test if the binary column has val_1 iff the other 2 columns are equal
            sample_okay_val_1 = False
            if (test_series_0.tolist().count(True) < 1) and (test_series_1.tolist().count(False) < 1):
                sample_okay_val_1 = True

            if not sample_okay_val_0 and not sample_okay_val_1:
                return

            # Test on the full columns
            # Todo: this handles when the dtype is 'category', but it may be faster to treat as category
            if sub_df_0[col_name_2].dtype.name == 'category' or sub_df_0[col_name_3].dtype.name == 'category':
                test_series_0 = sub_df_0[col_name_2].astype(str) == sub_df_0[col_name_3].astype(str)
                test_series_1 = sub_df_1[col_name_2].astype(str) == sub_df_1[col_name_3].astype(str)
            else:
                test_series_0 = sub_df_0[col_name_2] == sub_df_0[col_name_3]
                test_series_1 = sub_df_1[col_name_2] == sub_df_1[col_name_3]
            value_when_match = -1

            test_series = None

            # Test if the binary column has val_0 iff the other 2 columns are equal
            if (test_series_0.tolist().count(False) < self.contamination_level) and \
                    (test_series_1.tolist().count(True) < self.contamination_level):
                value_when_match = val0
                value_when_dont = val1
                test_series = [True] * self.num_rows
                for i in sub_df_0.index:
                    if not test_series_0[i]:
                        test_series[i] = False
                for i in sub_df_1.index:
                    if test_series_1[i]:
                        test_series[i] = False

            # Test if the binary column has val_1 iff the other 2 columns are equal
            if (test_series_0.tolist().count(True) < self.contamination_level) and \
                    (test_series_1.tolist().count(False) < self.contamination_level):
                value_when_match = val1
                value_when_dont = val0
                test_series = [True] * self.num_rows
                for i in sub_df_0.index:
                    if test_series_0[i]:
                        test_series[i] = False
                for i in sub_df_1.index:
                    if not test_series_1[i]:
                        test_series[i] = False

            if not test_series:
                return

            test_series = test_series | self.orig_df[bin_col].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_3].isna()

            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_2, col_name_3, bin_col]),
                [col_name_2, col_name_3, bin_col],
                test_series,
                (f"Column {bin_col} consistently has value {value_when_match} when columns {col_name_2} and "
                 f"{col_name_3} match values, and value {value_when_dont} when they do not match")
            )

        for bin_idx, bin_col in enumerate(self.binary_cols):
            if self.verbose >= 2:
                print(f"  Examining binary column {bin_idx} of {len(self.binary_cols)}")

            # Test the binary column contains both values a reasonable amount.
            val0, val1 = self.column_unique_vals[bin_col]
            if self.orig_df[bin_col].tolist().count(val0) < (self.num_rows * 0.1):
                continue
            if self.orig_df[bin_col].tolist().count(val1) < (self.num_rows * 0.1):
                continue

            # Create 2 dataframes that each cover one of the two values in the binary column. Do for the sample
            # dataframe as well.
            sample_sub_df_0 = self.sample_df[self.sample_df[bin_col] == val0]
            sample_sub_df_1 = self.sample_df[self.sample_df[bin_col] == val1]
            sub_df_0 = self.orig_df[self.orig_df[bin_col] == val0]
            sub_df_1 = self.orig_df[self.orig_df[bin_col] == val1]

            # Test for each pair columns, where the pair are of the same type.
            _, pairs = self.__get_numeric_column_pairs_unique()
            for col_name_2, col_name_3 in pairs:
                test_set(bin_col, col_name_2, col_name_3)

            for col_name_2, col_name_3 in self.__get_string_column_pairs_unique():
                test_set(bin_col, col_name_2, col_name_3)

            for col_name_2, col_name_3 in self.__get_date_column_pairs_unique():
                test_set(bin_col, col_name_2, col_name_3)

            for col_name_2, col_name_3 in self.__get_binary_column_pairs_unique():
                test_set(bin_col, col_name_2, col_name_3)

    #########################################################################
    # Data consistency checks for sets of three columns, where one is binary and the other two string
    #########################################################################
    def __generate_binary_two_str_match(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        # todo: think through how to code this. I think break into smaller, simple tests:
        # binary col true if: 1) the 2 string cols have the same length, but are many lengths in the columns;
        #   2) the 2 string cols have almost the same set of characters
        # self.__add_synthetic_column('bin_str_match rand_a', np.random.choice(['a', 'aa', 'aaa', 'aaaa'], self.num_synth_rows))
        # self.__add_synthetic_column('bin_str_match rand_b', np.random.choice(['a', 'aa', 'b', 'bb'], self.num_synth_rows))
        # self.__add_synthetic_column('bin_str_match all', self.synth_df['bin_str_match rand_a'] == self.synth_df['bin_str_match rand_b'])
        # self.__add_synthetic_column('bin_str_match most', self.synth_df['bin_str_match all'])

    def __check_binary_two_str_match(self, test_id):
        """
        """
        pass

    #########################################################################
    # Data consistency sets of multiple columns, where one is binary and the others are numeric
    #########################################################################
    def __generate_binary_matches_sum(self):
        """
        Patterns without exceptions:
        Patterns with exception: The test for the threshold is approximate, so rows near the threshold may be flogged.
        """
        self.__add_synthetic_column('bin_match_sum rand_a', np.random.randint(1, 1000, self.num_synth_rows))
        self.__add_synthetic_column('bin_match_sum rand_b', np.random.randint(1, 1000, self.num_synth_rows))
        self.__add_synthetic_column('bin_match_sum all',
            self.synth_df['bin_match_sum rand_a'] + self.synth_df['bin_match_sum rand_b'] > 1000)
        self.__add_synthetic_column('bin_match_sum most', self.synth_df['bin_match_sum all'])
        self.synth_df.loc[999, 'bin_match_sum most'] = not self.synth_df.loc[999, 'bin_match_sum most']

    def __check_binary_matches_sum(self, test_id):
        """
        """
        for bin_col in self.binary_cols:
            val0, val1 = self.column_unique_vals[bin_col]
            if self.orig_df[bin_col].tolist().count(val0) < (self.num_rows * 0.1):
                continue
            if self.orig_df[bin_col].tolist().count(val1) < (self.num_rows * 0.1):
                continue
            sub_df_0 = self.orig_df[self.orig_df[bin_col] == val0]
            sub_df_1 = self.orig_df[self.orig_df[bin_col] == val1]
            _, pairs = self.__get_numeric_column_pairs_unique()
            for num_col_1, num_col_2 in pairs:
                if not self.check_columns_same_scale_2(num_col_1, num_col_2):
                    continue

                sum_arr = pd.Series(self.orig_df[num_col_1] + self.orig_df[num_col_2])
                sum_arr_0 = sum_arr.loc[sub_df_0.index]
                sum_arr_1 = sum_arr.loc[sub_df_1.index]
                set_0_01_percentile = sum_arr_0.quantile(0.01)
                set_0_99_percentile = sum_arr_0.quantile(0.99)
                set_1_01_percentile = sum_arr_1.quantile(0.01)
                set_1_99_percentile = sum_arr_1.quantile(0.99)

                # Test if the binary column is consistently val_0 for the larger values in the numeric column
                if set_0_01_percentile > set_1_99_percentile:
                    threshold = statistics.mean(set_0_01_percentile, set_1_99_percentile)
                    test_series = [True if (x == val0 and y > threshold) or (x == val1 and y <= threshold) else False
                                   for x, y in zip(self.orig_df[bin_col], sum_arr)]
                    test_series = test_series | self.orig_df[bin_col].isna() | self.orig_df[num_col_1].isna() | self.orig_df[num_col_2].isna()
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([num_col_1, num_col_2, bin_col]),
                        [num_col_1, num_col_2, bin_col],
                        test_series,
                        (f'Column "{bin_col}" is consistently {val0} when the sum of columns "{num_col_1}" and '
                         f'"{num_col_2}" is over {threshold}'),
                    )

                # Test if the binary column is consistently val_1 for the larger values in the numeric column
                if set_1_01_percentile > set_0_99_percentile:
                    threshold = statistics.mean([set_1_01_percentile, set_0_99_percentile])
                    test_series = [True if (x == val1 and y > threshold) or (x == val0 and y <= threshold) else False
                                   for x, y in zip(self.orig_df[bin_col], sum_arr)]
                    test_series = test_series | self.orig_df[bin_col].isna() | self.orig_df[num_col_1].isna() | self.orig_df[num_col_2].isna()
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([num_col_1, num_col_2, bin_col]),
                        [num_col_1, num_col_2, bin_col],
                        test_series,
                        (f'Column "{bin_col}" is consistently {val1} when the sum of columns "{num_col_1}" and '
                         f'"{num_col_2}" is over {threshold}'),
                    )

    #########################################################################
    # Data consistency checks for single non-numeric columns
    #########################################################################
    def __generate_blank(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'blank_vals_most' has consistently non-blank values, with 2 exceptions
        """
        self.__add_synthetic_column('blank_vals_rand',
                                    ["aaaa"] * (self.num_synth_rows//2) + \
                                    [""] * (self.num_synth_rows//4) + \
                                    ["   "] * (self.num_synth_rows//4))
        self.__add_synthetic_column('blank_vals_most', ["aaaa"] * (self.num_synth_rows - 2) + [""] + ["   "])

    def __check_blank(self, test_id):
        for col_name in self.string_cols:
            test_series = (self.orig_df[col_name].astype(str) != "") & (~self.orig_df[col_name].astype(str).str.isspace())
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'Column "{col_name}" consistently contains non-blank values'
            )

    def __generate_leading_whitespace(self):
        """
        Patterns without exceptions: None. 'lead_white_all' has consistently zero leading white space, but this is
            not flagged as a pattern.
        Patterns with exception: 'lead_white_most_1' has consistently 0 leading whitespace characters, with 1 exception.
            'lead_white_most_2' has consistently 2 leading whitespace characters, with 2 exceptions
        """
        self.__add_synthetic_column('lead_white_rand',
                                    ["aaaaaaa"] * (self.num_synth_rows // 4) + \
                                    [" aaaaaa"] * (self.num_synth_rows // 4) + \
                                    ["  aaaaa"] * (self.num_synth_rows // 4) + \
                                    ["   aaaa"] * (self.num_synth_rows // 4))
        self.__add_synthetic_column('lead_white_all', ["a"] * (self.num_synth_rows - 2) + ["aa"] + ["aaa"])
        self.__add_synthetic_column('lead_white_most_1', ["aaaa"] * (self.num_synth_rows - 2) + ["      aaa"] + ["       aaa"])
        self.__add_synthetic_column('lead_white_most_2', ["  aaaa"] * (self.num_synth_rows - 2) + ["aaa"] + ["      aaa"])

    def __check_leading_whitespace(self, test_id):
        """
        This does not report columns with consistently zero leading whitespace characters as a pattern, as this is
        normally the case and not considered interesting.
        """

        for col_name in self.string_cols:
            test_series_counts = self.orig_df[col_name].astype(str).str.len() - \
                                 self.orig_df[col_name].astype(str).str.lstrip().str.len()
            if test_series_counts.max() == 0:
                continue

            test_series_non_null_counts = pd.Series([
                (x-y)
                for w, x, y in zip(self.orig_df[col_name],
                                   self.orig_df[col_name].astype(str).str.len(),
                                   self.orig_df[col_name].astype(str).str.lstrip().str.len())
                if (not is_missing(w))])

            # Test if there are normally zero leading spaces
            test_series = self.orig_df[col_name].isna() | (test_series_counts == 0)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'Column "{col_name}" consistently contains values without leading spaces'
            )

            # Test if there are a normal number of trailing spaces
            median_num_spaces = test_series_non_null_counts.median()
            test_series = self.orig_df[col_name].apply(is_missing) | \
                          ((test_series_counts > (median_num_spaces / 2)) & \
                           (test_series_counts < (median_num_spaces * 2)))
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f'Column "{col_name}" consistently contains about {median_num_spaces} leading spaces (minimum: '
                 f'{test_series_counts.min()}, maximum: {test_series_counts.max()}'),
                'with significantly more or less leading spaces'
            )

    def __generate_trailing_whitespace(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'trail_white_most' has consistently 2 trailing whitespace characters, with 2 exceptions
        """
        self.__add_synthetic_column('trail_white_rand',
                                    ["aaaaaaa"] * (self.num_synth_rows // 4) + \
                                    ["aaaaaa "] * (self.num_synth_rows // 4) + \
                                    ["aaaaa  "] * (self.num_synth_rows // 4)+ \
                                    ["aaaa   "] * (self.num_synth_rows // 4))
        self.__add_synthetic_column('trail_white_all', ["a"] * (self.num_synth_rows - 2) + ["aa"] + ["aaa"])
        self.__add_synthetic_column('trail_white_most', ["aaaa  "] * (self.num_synth_rows - 2) + ["aaa"] + ["aaa      "])

    def __check_trailing_whitespace(self, test_id):
        for col_name in self.string_cols:
            test_series_counts = self.orig_df[col_name].astype(str).str.len() - self.orig_df[col_name].astype(str).str.rstrip().str.len()
            if test_series_counts.max() == 0:
                continue
            test_series_non_null_counts = pd.Series([
                (x-y)
                for w, x, y in zip(self.orig_df[col_name],
                                   self.orig_df[col_name].astype(str).str.len(),
                                   self.orig_df[col_name].astype(str).str.lstrip().str.len())
                if (not is_missing(w))])

            # Test if there are normally zero trailing spaces
            test_series = self.orig_df[col_name].isna() | (test_series_counts == 0)
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'Column "{col_name}" consistently contains values without trailing spaces'
            )

            # Test if there are a normal number of trailing spaces
            median_num_spaces = test_series_non_null_counts.median()
            test_series = self.orig_df[col_name].apply(is_missing) | \
                          ((test_series_counts > (median_num_spaces / 2)) & \
                           (test_series_counts < (median_num_spaces * 2)))
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f'Column "{col_name}" consistently contains about {median_num_spaces} trailing spaces (minimum: '
                 f'{test_series.min()}, maximum: {test_series.max()}'),
                'with significantly more or less trailing spaces'
            )

    def __generate_first_char_alpha(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('check first char alpha rand',
            [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('check first char alpha all ',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('check first char alpha most',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['8abcd'])

    def __check_first_char_alpha(self, test_id):
        for col_name in self.string_cols:
            # Skip columns that have only one character for all values
            # todo: fill in

            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.lstrip().str.slice(0, 1).str.isalpha()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.lstrip().str.slice(0, 1).str.isalpha()
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently starts with an alphabetic character")

    def __generate_first_char_numeric(self):
        """
        Patterns without exceptions: 'first_char_numeric_2' consistently begins with a digit
        Patterns with exception: 'first_char_numeric_3' consistently begins with a digit, with 1 exception
        """
        self.__add_synthetic_column('first_char_numeric_1', [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_char_numeric_2', [[''.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_char_numeric_3', [[''.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])

    def __check_first_char_numeric(self, test_id):
        for col_name in self.string_cols:
            # Skip columns that have only one character for all values
            # todo: fill in

            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.slice(0, 1).str.isdigit()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.slice(0, 1).str.isdigit()
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently starts with a numeric character")

    def __generate_first_char_small_set(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('first char small set rand', [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first char small set all ', [[''.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first char small set most', [[''.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['xabcde'])

    def __check_first_char_small_set(self, test_id):
        for col_name in self.string_cols:
            # todo: Skip columns that have only one character for all values

            test_series = self.orig_df[col_name].astype(str).str.slice(0, 1)  # Get the first letter of each value

            # Skip columns that have almost as many distinct values as distinct first characters
            if test_series.nunique() > (self.orig_df[col_name].nunique() / 2):
                continue

            counts_series = test_series.value_counts(normalize=False) # Get the counts for each first letter
            # Check if there's a small set of characters that make up the bulk of the values
            count_most_common = np.where(counts_series.sort_values(ascending=False).cumsum() > (self.num_rows - self.contamination_level))[0][0]
            if count_most_common <= 5:
                common_first_chars = list(counts_series.sort_values(ascending=False)[:count_most_common+1].index)
                rare_first_chars = []
                for c in counts_series.index:
                    if c not in common_first_chars:
                        rare_first_chars.append(c)
                results_col = test_series.isin(common_first_chars)
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    results_col,
                    (f'The column "{col_name}" has {self.orig_df[col_name].nunique()} distinct values and contains '
                     f'values that consistently start with one of {common_first_chars}'),
                    f'with some values starting with one of {rare_first_chars}'
                )

    def __generate_first_char_uppercase(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('first_char_upper rand', [[''.join(random.choice(alphanumeric)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_char_upper all ', [[''.join(random.choice(['A', 'B', 'C'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_char_upper most', [[''.join(random.choice(['A', 'B', 'C'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])

    def __check_first_char_uppercase(self, test_id):
        for col_name in self.string_cols:
            # Skip columns where there is only one character used for the first character in all values
            first_chars = self.orig_df[col_name].astype(str).str[:1]
            if first_chars.nunique() == 1:
                continue

            # todo: flags A with accent
            # todo: don't flag None or NaN
            test_series = self.orig_df[col_name].astype(str).str[:1].isin(list(string.ascii_uppercase))
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently starts with an uppercase character")

    def __generate_first_char_lowercase(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('first_char_lower rand', [[''.join(random.choice(alphanumeric)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_char_lower all ', [[''.join(random.choice(['a', 'b', 'c'])
             for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_char_lower most', [[''.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['Abcde'])

    def __check_first_char_lowercase(self, test_id):
        for col_name in self.string_cols:
            # Skip columns where there is only one character used for the first character in all values
            first_chars = self.orig_df[col_name].astype(str).str[:1]
            if first_chars.nunique() == 1:
                continue

            test_series = self.orig_df[col_name].astype(str).str[:1].isin(list(string.ascii_lowercase))
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column consistently starts with an lowercase character")

    def __generate_last_char_small_set(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('last char small set rand', [[''.join(random.choice(alphanumeric)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('last char small set all ', [[''.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('last char small set most', [[''.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])

    def __check_last_char_small_set(self, test_id):
        for col_name in self.string_cols:
            test_series = self.orig_df[col_name].astype(str).str[-1:]
            counts_series = test_series.value_counts(normalize=False)
            # Check if there's a small set of character that make up the bulk of the values
            count_most_common = np.where(counts_series.sort_values(ascending=False).cumsum() > (self.num_rows - self.contamination_level))[0][0]
            if count_most_common <= 5:
                common_last_chars = list(counts_series.sort_values(ascending=False)[:count_most_common+1].index)
                rare_last_chars = []
                for c in counts_series.index:
                    if c not in common_last_chars:
                        rare_last_chars.append(c)
                results_col = test_series.isin(common_last_chars)
                self.__process_analysis_binary(
                    test_id,
                    col_name,
                    [col_name],
                    results_col,
                    (f'The column "{col_name}" has {self.orig_df[col_name].nunique()} distinct values and contains '
                     f'values that consistently end with one of {common_last_chars}'),
                    f'with some values ending with one of {rare_last_chars}'
                )

    def __generate_common_special_chars(self):
        """
        Patterns without exceptions: 'common_spec_chars all' consistently contains '*'
        Patterns with exception: 'common_spec_chars most' does as well, with 1 exception
        """
        self.__add_synthetic_column('common_spec_chars rand', [[''.join(random.choice(list(alphanumeric) + ['*','&', '#'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('common_spec_chars all',  [['*'.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('common_spec_chars most', [['*'.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])

    def __check_common_special_chars(self, test_id):

        def get_non_alphanumeric(x):
            if x.isalnum():
                return []
            return [c for c in x if not str(c).isalnum()]

        for col_name in self.string_cols:
            # todo: Skip columns that have only one character for all values

            # Skip columns which contain only alphanumeric characters
            if self.orig_df[col_name].astype(str).str.isalnum().tolist().count(False) == 0:
                continue

            # Get the set of special characters in each value
            special_chars_list = self.orig_df[col_name].astype(str).apply(get_non_alphanumeric)

            # Remove any empty lists
            special_chars_list = [x for x in special_chars_list if len(x)]

            # Skip columns where it is not the case that almost all values have special characters
            if (len(special_chars_list) + self.orig_df[col_name].isna().sum()) < (self.num_rows - self.contamination_level):
                continue

            # Get the unique set of special characters
            special_chars_list = list(set([item for sublist in special_chars_list for item in sublist]))

            # Examine each special character and determine if it is in most values
            common_special_chars_list = []
            for c in special_chars_list:
                if (self.orig_df[col_name].isna().sum() +
                    self.orig_df[col_name].astype(str).str.contains(c, regex=False).tolist().count(True)) > \
                        (self.num_rows - self.contamination_level):
                    common_special_chars_list.append(c)

            if len(common_special_chars_list) == 0:
                continue

            # Having a space in each record is not interesting
            if common_special_chars_list == [' ']:
                continue

            # Identify the rows that do not contain all the common special characters
            test_series = [True] * self.num_rows
            for c in common_special_chars_list:
                test_series = test_series & self.orig_df[col_name].astype(str).str.contains(c, regex=False)
            test_series = test_series | self.orig_df[col_name].isna()

            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f"The column consistently contains the {common_special_chars_list} character(s)",
                allow_patterns=True)

    # todo: rename to common_chars
    def __generate_count_repeated(self):
        """
        Patterns without exceptions: 'repeated_chars all ' contains an 'x' in all values.
        Patterns with exception: 'repeated_chars most' contains an 'x' in most values, but not the last
        """
        self.__add_synthetic_column('repeated_chars rand', [[''.join(random.choice(list(alphanumeric))
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('repeated_chars all ', [['x'.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('repeated_chars most', [['x'.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])

    def __check_count_repeated(self, test_id):
        def get_unique_chars(x):
            return list(set(x))

        for col_idx, col_name in enumerate(self.string_cols):
            if self.verbose >= 2:
                print(f"  Processing column {col_idx} of {len(self.string_cols)} string columns")

            # Skip columns that have mostly a single character
            value_lens_arr = pd.Series(self.orig_df[col_name].astype(str).str.len())
            if value_lens_arr.quantile(0.9) <= 1:
                continue

            # Test on a sample
            # Get the set of special characters in each value
            sample_unique_chars_list = self.sample_df[col_name].dropna().astype(str).apply(get_unique_chars)

            # Get the unique set of special characters
            sample_unique_chars_list = list(set([item for sublist in sample_unique_chars_list for item in sublist]))

            # Examine each special character and determine if it is in most values
            common_chars_list = []
            for c in sample_unique_chars_list:
                if (self.sample_df[col_name].isna().sum() + \
                    self.sample_df[col_name].fillna("").astype(str).str.contains(c, regex=False).tolist().count(False)) <= 1:
                    common_chars_list.append(c)
            if len(common_chars_list) == 0:
                continue

            test_series = [True] * len(self.sample_df)
            for c in common_chars_list:
                test_series = test_series & self.sample_df[col_name].astype(str).str.contains(c, regex=False)
            test_series = test_series | (self.sample_df[col_name].isna())
            # Give some wiggle room, as the common_chars_list is based on a small sample
            if test_series.tolist().count(False) > 5:
                continue

            # Test on the full column

            # Get the set of special characters in each value
            unique_chars_list = self.orig_df[col_name].dropna().astype(str).apply(get_unique_chars)

            # Get the unique set of special characters
            unique_chars_list = list(set([item for sublist in unique_chars_list for item in sublist]))

            # Examine each special character and determine if it is in most values
            common_chars_list = []
            for c in unique_chars_list:
                if (self.orig_df[col_name].isna().sum() + \
                        self.orig_df[col_name].fillna("").astype(str).str.contains(c, regex=False).tolist().count(True)) > \
                    (self.num_rows - self.contamination_level):
                    common_chars_list.append(c)

            if len(common_chars_list) == 0:
                continue

            # Identify the rows that do not contain all the common special characters
            test_series = [True] * self.num_rows
            for c in common_chars_list:
                test_series = test_series & self.orig_df[col_name].astype(str).str.contains(c, regex=False)
            test_series = test_series | (self.orig_df[col_name].isnull())

            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f"The column consistently contains the all of the following characters: {common_chars_list}",
                allow_patterns=True)

    def __generate_number_alpha_chars(self):
        """
        Patterns without exceptions: 'number alpha 2' consistently has 5 alphabetic characters
        Patterns with exception: 'number alpha 3 does as well, wiht 1 exception
        """
        self.__add_synthetic_column('number alpha 1',
            [[''.join(random.choice(letters)
                      for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('number alpha 2',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('number alpha 3',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcdef'])

    def __check_number_alpha_chars(self, test_id):
        # todo: all these tests, don't flag None or NaN.
        for col_name in self.string_cols:
            test_series = self.orig_df[col_name].fillna("").astype(str).apply(lambda x: len([e for e in x if e.isalpha()]))
            self.__process_analysis_counts(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains values with consistently", "alphabetic characters")

    def __generate_number_numeric_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column(
            'number numeric 1',
            [[''.join(random.choice(digits) for _ in range(random.randint(1,10)))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'number numeric 2',
            [[''.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'number numeric 3',
            [[''.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['432'])

    def __check_number_numeric_chars(self, test_id):
        for col_name in self.string_cols:
            test_series = self.orig_df[col_name].astype(str).apply(lambda x: len([e for e in x if e.isdigit()]))
            self.__process_analysis_counts(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains values with consistently", " numeric characters")

    def __generate_number_alphanumeric_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('number alphanumeric 1', [['$'.join(random.choice(digits)
            for _ in range(random.randint(1,10)))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('number alphanumeric 2', [['$'.join(random.choice(digits)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('number alphanumeric 3', [['$'.join(random.choice(digits)
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['$432'])

    def __check_number_alphanumeric_chars(self, test_id):
        for col_name in self.string_cols:
            test_series = self.orig_df[col_name].fillna("").astype(str).apply(lambda x: len([e for e in x if e.isalnum()]))
            test_series = [test_series.median() if y else x for x, y in zip(test_series, self.orig_df[col_name].isnull())]
            self.__process_analysis_counts(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains values with consistently", " alpha-numeric characters")

    def __generate_number_non_alphanumeric_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('non-alphanumeric rand', [['@'.join(random.choice(digits) for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('non-alphanumeric all ', [['@'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('non-alphanumeric most', [['@'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['$432'])

    def __check_number_non_alphanumeric_chars(self, test_id):
        for col_name in self.string_cols:
            test_series = self.orig_df[col_name].astype(str).apply(lambda x: len([e for e in x if not e.isalnum()]))
            self.__process_analysis_counts(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains values with consistently", " non-alphanumeric characters")

    def __generate_number_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('num_chars rand', [[''.join(random.choice(digits) for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('num_chars all ', [[''.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('num_chars most', [[''.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['$432'])

    def __check_number_chars(self, test_id):
        for col_name in self.string_cols:
            test_series = self.orig_df[col_name].fillna("").astype(str).str.len()
            self.__process_analysis_counts(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column contains values with consistently", " characters")

    def __generate_many_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column(
            'many_chars all',
            [''.join(['a' for _ in range(np.random.randint(2,20))]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'many_chars most',
            [''.join(['a' for _ in range(np.random.randint(2,20))]) for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'many_chars most'] = ''.join(['a']*100)

    def __check_many_chars(self, test_id):
        for col_name in self.string_cols:
            val_lens = self.orig_df[col_name].astype(str).str.len()
            q1 = val_lens.quantile(0.25)
            q3 = val_lens.quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))
            test_series = val_lens < upper_limit
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f"Some values were unusually long. Any values with length greater than {upper_limit} characters "
                 f"were flagged, given the 25th percentile of string length is {q1} and the 75th is {q3}"),
                allow_patterns=False
            )

    def __generate_few_chars(self):
        """
        Patterns without exceptions: None: 'few_chars all' consistently has 100 to 200 characters, but this test is
            not flagged as a pattern.
        Patterns with exception: 'few_chars most' is similar, but has 1 exception with only 2 characters.
        """
        self.__add_synthetic_column('few_chars all', [''.join(['a' for _ in range(np.random.randint(100, 200))]) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('few_chars most', [''.join(['a' for _ in range(np.random.randint(100, 200))]) for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'few_chars most'] = 'aa'

    def __check_few_chars(self, test_id):
        """
        For this, we use the inter-decile range instead of inter-quartile, as IQR often sets the lower limit below zero
        which is impossible here, while small positive values may nevertheless be unusually small.
        """

        for col_name in self.string_cols:
            val_lens = self.orig_df[col_name].fillna("").astype(str).str.len()
            val_lens_no_null = self.orig_df[col_name].dropna().astype(str).str.len()
            d1 = val_lens_no_null.quantile(0.1)
            d9 = val_lens_no_null.quantile(0.9)
            lower_limit = d1 - (self.idr_limit * (d9 - d1))
            test_series = val_lens > lower_limit
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f"Some values were unusually short. Any values with length less than {lower_limit} characters "
                 f"were flagged, given the 10th percentile of string length is {d1} and the 90th is {d9}. "
                 f"The coefficient is set at {self.idr_limit}"),
                allow_patterns=False
            )

    def __generate_position_non_alphanumeric(self):
        """
        Patterns without exceptions: 'posn-special all' consistently has a '-' in the 3rd position
        Patterns with exception: 'posn-special most' is similar, but with 1 exception
        """
        self.__add_synthetic_column(
            'posn-special rand',
            [[''.join(random.choice(list(letters) + ['-']) for _ in range(random.randint(1, 10)))][0]
             for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'posn-special all',
            [['ab-' + ''.join(random.choice(letters) for _ in range(5))][0]
             for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column(
            'posn-special most',
            [['ab-' + ''.join(random.choice(letters) for _ in range(5))][0]
             for _ in range(self.num_synth_rows-1)] + ['ab$43-2'])

    def __check_position_non_alphanumeric(self, test_id):
        """
        Check if the position of the non-alphanumeric columns is consistent. For example, a column may have values
        such as 'AB-123', 'WT-15445' etc. Here the hyphen in consistently in the 2nd (zero-based) position. We check,
        where the strings are not a consistent length, both the position from the front and from the back of the
        strings.

        The test COMMON_SPECIAL_CHARS checks if there are any special characters present in most values. This test
        checks, for any special characters that are in all values, if they are in the same position.
        """
        def get_non_alphanumeric(x):
            if x.isalnum():
                return []
            return [c for c in x if not str(c).isalnum()]

        for col_name in self.string_cols:
            # todo: Skip columns that have only one character for all values

            # Skip columns which contain only alphanumeric characters
            if self.orig_df[col_name].astype(str).str.isalnum().tolist().count(False) == 0:
                continue

            # Get the set of special characters in each value
            special_chars_list = self.orig_df[col_name].astype(str).apply(get_non_alphanumeric)

            # Remove any empty lists
            special_chars_list = [x for x in special_chars_list if len(x)]

            # Skip columns where it is not the case that all values have special characters
            if (self.orig_df[col_name].isna().sum() + len(special_chars_list)) < self.num_rows:
                continue

            # Get the unique set of special characters
            special_chars_list = list(set([item for sublist in special_chars_list for item in sublist]))

            # Examine each special character and determine if it is in most values
            common_special_chars_list = []
            for c in special_chars_list:
                if (self.orig_df[col_name].isna().sum() + \
                        self.orig_df[col_name].fillna("").astype(str).str.contains(c, regex=False).tolist().count(True)) > \
                        (self.num_rows - self.contamination_level):
                    common_special_chars_list.append(c)

            if len(common_special_chars_list) == 0:
                continue

            # Identify the common special characters which appear in a consistent location
            for c in common_special_chars_list:
                # Get the position of the character from the beginning of the strings
                list_1 = pd.Series(self.orig_df[col_name].astype(str).str.find(c)).replace(-1, np.NaN)
                # Get the position of the character from the end of the strings
                list_2 = pd.Series([x - y if y >= 0 else -1 for x, y in
                                    zip(self.orig_df[col_name].fillna("").astype(str).str.len() ,
                                        self.orig_df[col_name].fillna("").astype(str).str.rfind(c))]).replace(-1, np.NaN)
                vc1 = list_1.value_counts()
                vc2 = list_2.value_counts()
                if (self.orig_df[col_name].isna().sum() + vc1.iloc[0]) >= (self.num_rows - self.contamination_level):
                    common_posn = vc1.index[0]
                    common_posn_str = str(common_posn)
                    test_series = self.orig_df[col_name].astype(str).str.find(c) == common_posn
                elif (self.orig_df[col_name].isna().sum() + vc2.iloc[0]) >= (self.num_rows - self.contamination_level):
                    common_posn = vc2.index[0]
                    common_posn_str = f"{common_posn} from the end"
                    test_series = (self.orig_df[col_name].astype(str).str.len() -
                                   self.orig_df[col_name].astype(str).str.rfind(c)) == common_posn
                else:
                    continue
                test_series = test_series | self.orig_df[col_name].isna()

                pattern_str = f"The column contains values consistently with '{c}' in position {common_posn_str}"
                if test_series.tolist().count(False) == 0:
                    self.patterns_arr.append([test_id, col_name, pattern_str, None])

                elif test_series.tolist().count(False) < self.contamination_level:
                    self.__process_analysis_binary(
                        test_id,
                        col_name,
                        [col_name],
                        test_series,
                        pattern_str
                    )
                    break

    def __generate_uppercase(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('uppercase rand', [[''.join(random.choice(string.ascii_letters)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('uppercase all ', [[''.join(random.choice(string.ascii_uppercase)
            for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('uppercase most', [[''.join(random.choice(string.ascii_uppercase)
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['YYa'])

    def __check_uppercase(self, test_id):
        for col_name in self.string_cols:
            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.isupper()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.isupper()
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column is consistently uppercase")

    def __generate_lowercase(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('lowercase rand', [[''.join(random.choice(string.ascii_letters)
            for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('lowercase all ', [[''.join(random.choice(string.ascii_lowercase)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('lowercase most', [[''.join(random.choice(string.ascii_lowercase)
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['YYa'])

    def __check_lowercase(self, test_id):
        for col_name in self.string_cols:
            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.islower()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.islower()
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                "The column is consistently lowercase")

    def __generate_characters_used(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('char counts rand', [[''.join(random.choice(string.ascii_letters)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('char counts all ', [[''.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('char counts most', [[''.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['Ybbca'])

    # todo: is the same as COMMON_CHARS (once renamed)??
    # todo: it doesn't call process_binary or count()
    def __check_characters_used(self, test_id):
        for col_name in self.string_cols:
            if self.orig_df[col_name].astype(str).str.len().max() > 5:
                continue
            full_text = ''.join(list(self.orig_df[col_name].astype(str)))
            rare_chars = {}
            common_chars = {}
            for c in string.ascii_letters:
                count_c = full_text.count(c)
                if count_c == 0:
                    continue
                if count_c < len(full_text) / 1000:
                    rare_chars[c] = count_c
                else:
                    common_chars[c] = count_c

            if (0 < len(common_chars) <= 5) and (len(rare_chars) == 0):
                self.patterns_arr.append([
                    test_id,
                    col_name,
                    f'Column: "{col_name}" contains only the characters: {list(common_chars.keys())}',
                    None])
            if len(common_chars) > 0 and 0 < len(rare_chars) < 10:
                results_col_name = self.get_results_col_name(test_id, col_name)
                results_col = [0] * self.num_rows
                for rare_char in rare_chars.keys():
                    results_col = results_col | self.orig_df[col_name].str.contains(rare_char)
                self.__add_result_column(results_col_name, results_col, [col_name])
                # todo: this does not call __process_binary
                self.results_summary_arr.append([
                    test_id,
                    col_name,
                    f'The column has mostly values with {list(common_chars.keys())[:5]}... ({len(common_chars)} unique common values), with exceptions with {list(rare_chars.keys())}.',
                    results_col.tolist().count(True),
                    None])

    def __generate_first_word(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('first_word rand', [''.join(np.random.choice(['a', 'b' ' '], 10)) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_word all', ["abc-" + np.random.choice(list(string.ascii_letters)) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('first_word most', self.synth_df['first_word all'])
        self.synth_df.loc[999, 'first_word most'] = "wyxz"

    def __check_first_word(self, test_id):
        """
        Here we define words as strings separated by whitespace or common separator characters such as hyphens.
        """
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)

            # Skip columns that are primarily 1 word
            word_arr = col_vals.str.split()
            num_words_arr = [len(x) for x in word_arr]
            if pd.Series(num_words_arr).quantile(0.9) <= 1:
                continue

            first_words = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            vc = pd.Series(first_words).value_counts()
            common_values = []
            for v in pd.Series(first_words).unique():
                if vc[v] > self.contamination_level:
                    common_values.append(v)
            if len(common_values) > self.contamination_level:
                continue
            test_series = [x in common_values for x in first_words]
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                np.array(test_series),
                f"The column consistently begins with one of {common_values}")

    def __generate_last_word(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('last_word rand', [''.join(np.random.choice(['a', 'b' ' '], 10)) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('last_word all', [np.random.choice(list(string.ascii_letters)) + "-abc" for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('last_word most', self.synth_df['last_word all'])
        self.synth_df.loc[999, 'last_word most'] = "wyxz"

    def __check_last_word(self, test_id):
        """
        Here we define words as strings separated by whitespace or common separator characters such as hyphens.
        """
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)

            # Skip columns that are primarily 1 word
            word_arr = col_vals.str.split()
            num_words_arr = [len(x) for x in word_arr]
            if pd.Series(num_words_arr).quantile(0.1) <= 1:
                continue

            last_words = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
            vc = pd.Series(last_words).value_counts()
            common_values = []
            for v in pd.Series(last_words).unique():
                if vc[v] > self.contamination_level:
                    common_values.append(v)
            if len(common_values) > self.contamination_level:
                continue
            test_series = [x in common_values for x in last_words]
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                np.array(test_series),
                f"The column consistently ends with one of {common_values}")

    def __generate_num_words(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('num_words all',
                                    [[''.join(random.choice(string.ascii_letters[:10] + ' ')
                                              for _ in range(np.random.randint(1, 20)))][0]
                                     for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('num_words most', self.synth_df['num_words all'])
        self.synth_df.loc[999, 'num_words all'] = 'a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c '
        pass

    def __check_num_words(self, test_id):
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            word_counts_arr = pd.Series([len(x) for x in word_arr])
            q1 = word_counts_arr.quantile(0.25)
            q3 = word_counts_arr.quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))
            test_series = [x < upper_limit for x in word_counts_arr]
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f'The values in column "{col_name}" have word count with 25th percentile {q1} characters and 75th '
                 f'percentile {q3} words'),
                f' Flagging any values with over {upper_limit} words.',
                allow_patterns=False
            )

    def __generate_longest_words(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('long_word all',
                                    [[''.join(random.choice(string.ascii_letters)
                                              for _ in range(np.random.randint(1, 12)))][0]
                                     for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('long_word most',
                                    [[''.join(random.choice(string.ascii_letters)
                                              for _ in range(np.random.randint(1, 12)))][0]
                                     for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'long_word most'] = 'ad dfddfdabdkdjsdlksjklsdjkldjklsdsjkldjklsdjklsd yy'
        pass

    def __check_longest_words(self, test_id):
        """
        This checks for invalid words within string, possibly due to missing white space between the words or otherwise
        invalid text content.
        """
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            word_lens_arr = [[len(w) for w in x] for x in word_arr]
            flat_word_lens_arr = pd.Series(np.concatenate(word_lens_arr).flat)
            q1 = flat_word_lens_arr.quantile(0.25)
            q3 = flat_word_lens_arr.quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * 1.5 * (q3 - q1))  # We use more than the normal limit
            max_word_lens = [max(x) if len(x) > 0 else 0 for x in word_lens_arr]
            test_series = [x < upper_limit for x in max_word_lens]
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                (f'The words in column "{col_name}" have lengths with 25th percentile {q1} characters and 75th '
                 f'percentile {q3} characters'),
                f' Flagging any values with words over {upper_limit} characters.',
                allow_patterns=False
            )

    def __generate_words_used(self):
            """
            Patterns without exceptions: 'words_used all' consistently has the word 'giblet'
            Patterns with exception: 'words_used most' consistently has the word 'giblet', with 1 exception
            """
            self.__add_synthetic_column('words_used rand_a', np.random.choice(["aa", "bb", "cc", "dd"], self.num_synth_rows))
            self.__add_synthetic_column('words_used rand_b', [[''.join(random.choice(string.ascii_letters)
                                                                       for _ in range(5))][0] for _ in range(self.num_synth_rows)])
            self.__add_synthetic_column('words_used all', "giblet " + self.synth_df['words_used rand_a'] + "-" + self.synth_df['words_used rand_b'])
            self.__add_synthetic_column('words_used most', self.synth_df['words_used all'])
            self.synth_df.loc[999, 'words_used most'] = "wyxz"

    def __check_words_used(self, test_id):
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            words_arr = col_vals.str.split()
            flat_words_arr = list(np.concatenate(words_arr).flat)
            vc = pd.Series(flat_words_arr).value_counts(ascending=False)
            common_words_arr = []
            for v in vc.index:
                if (self.orig_df[col_name].isna().sum() + vc[v]) >= (self.num_rows - self.contamination_level):
                    common_words_arr.append(v)
                else:
                    break
            if len(common_words_arr) == 0:
                continue
            test_series = [True] * self.num_rows
            for c in common_words_arr:
                test_series = test_series & np.array([c in words_arr[i] for i in range(self.num_rows)])
            test_series = test_series | self.orig_df[col_name].isna()
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The values in column "{col_name}" consistently contain the words {common_words_arr}'
            )

    def __generate_rare_words(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'rare_words most' consitently contains non-rare words, with one exception
        """
        word_list = [''.join([random.choice(string.ascii_letters) for x in range(10)]) for _ in range(50)]
        self.__add_synthetic_column('rare_words all', [' '.join([word_list[np.random.randint(0, len(word_list))]] * 4) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('rare_words most', [' '.join([word_list[np.random.randint(0, len(word_list))]] * 4) for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'rare_words most'] = self.synth_df.loc[999, 'rare_words most'] + " xxxxxxx"

    def __check_rare_words(self, test_id):
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            words_arr = col_vals.str.split()
            flat_words_arr = list(np.concatenate(words_arr).flat)
            vc = pd.Series(flat_words_arr).value_counts(ascending=True)
            rare_words_arr = []
            for v in vc.index:
                if vc[v] < self.contamination_level:
                    rare_words_arr.append(v)
                else:
                    break
            if len(rare_words_arr) == 0:
                continue
            if len(rare_words_arr) > self.contamination_level:
                continue
            test_series = [len(set(x).intersection(set(rare_words_arr))) == 0 for x in words_arr]
            self.__process_analysis_binary(
                test_id,
                col_name,
                [col_name],
                test_series,
                f'The values in column "{col_name}" consistently use common words ',
                f'Flagging values using the rare words {array_to_str(rare_words_arr)}'
            )

    def __generate_grouped_strings(self):
        """
        Patterns without exceptions: 'grouped_str all ' has the 3 values in this column in 3 consistent groups
        Patterns with exception: 'grouped_str most' is similar, with 1 exception
        """
        self.__add_synthetic_column('grouped_str rand', [random.choice(['aaa', 'bbbb', 'cccc'])
                                                         for _ in range(self.num_synth_rows)])
        group_a_len = group_b_len = self.num_synth_rows // 3
        group_c_len = self.num_synth_rows - (group_a_len + group_b_len)
        self.__add_synthetic_column('grouped_str all ', ['aaa'] * group_a_len +
                                                        ['bbb'] * group_b_len +
                                                        ['ccc'] * group_c_len)
        self.__add_synthetic_column('grouped_str most', ['aaa'] * group_a_len +
                                                        ['bbb'] * group_b_len +
                                                        ['ccc'] * (group_c_len - 1) + ['aaa'])

    def __check_grouped_stings_column(self, test_id, col_name, col_values):
        """
        This does not currently work well with many Null values.
        """

        col_df = pd.DataFrame({col_name: col_values})
        col_df['Next'] = col_df[col_name].shift(1)
        col_df['Same'] = col_df[col_name] == col_df['Next']
        num_same = col_df['Same'].tolist().count(True)
        # There will always be rows not like the next: where the list moves from one value to the next. So ideally,
        # the number of rows that are the same as the next is the total number of rows - (number values -1). As well,
        # the lastrow is always unlike the next, as the next is undefined.
        ideal_same = self.num_valid_rows[col_name] - self.orig_df[col_name].nunique()
        if (ideal_same - num_same) > self.contamination_level:
            return

        # todo: as with ascending & descending, in the detailed description, show the rows above & below
        test_series = np.array([1]*self.num_rows)
        if num_same != ideal_same:
            # Loop through each unique value and find the indexes where it occurs.
            # We then identify the runs of each unique value and flag any short runs.
            for v in col_df[col_name].dropna().unique():
                idxs = np.where(col_df[col_name] == v)[0]
                idx_diffs = pd.Series(idxs).shift(-1).values - idxs
                exceptions = list(np.where(idx_diffs != 1)[0])
                group_starts = [min(idxs)]
                group_ends = [max(idxs)]
                for e in exceptions:
                    group_starts.append(idxs[e] + idx_diffs[e])
                    group_ends.insert(0, idxs[e])

                group_starts = sorted([x for x in (set(group_starts)) if x == x])
                group_ends = sorted([x for x in (set(group_ends)) if x == x])

                for i in range(len(group_starts)):
                    group_len = group_ends[i] - group_starts[i] + 1
                    if group_len < (len(idxs) * 0.5):
                        for j in range(int(group_starts[i]), int(group_ends[i]+1)):
                            test_series[j] = False

        self.__process_analysis_binary(
            test_id,
            col_name,
            [col_name],
            test_series,
            f"The values in {col_name} are consistently grouped together"
        )

    def __check_grouped_strings(self, test_id):
        """
        This is similar to GROUPED_STRINGS_BY_NUMERIC, but uses the table order the data is received in.
        """
        for col_name in self.string_cols + self.binary_cols:
            if self.orig_df[col_name].nunique() > math.sqrt(self.num_rows):
                continue
            df = self.orig_df.copy()
            col_series = df[col_name]
            self.__check_grouped_stings_column(test_id, col_name, col_series)

    ####################################################################################
    # Data consistency checks for pairs of non-numeric columns
    ####################################################################################

    def __generate_a_implies_b(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        pass

    def __check_a_implies_b(self, test_id):
        pass

    def __generate_rare_pairs(self):
        """
        Patterns without exceptions: None: rand_c and rand_d columns appear with all combinations of values with the
            other columns, but this test does not flag patterns, only exceptions.
        Patterns with exception: the columns rand_a and rand_b have only 1 instance of 'b' and 'b'
        """
        vals = [
            ['a', 'a', 'x', 'x'],
            ['a', 'b', 'x', 'y'],
            ['b', 'a', 'y', 'x'],
            ['a', 'b', 'y', 'y'],
            ['a', 'b', 'x', 'y']]
        data = np.array([vals[np.random.choice(range(len(vals)))] for _ in range(self.num_synth_rows-1)])
        data = np.vstack([data, ['b', 'b', 'x', 'y']])
        self.__add_synthetic_column('rare_pair rand_a', data[:, 0])
        self.__add_synthetic_column('rare_pair rand_b', data[:, 1])
        self.__add_synthetic_column('rare_pair rand_c', data[:, 2])
        self.__add_synthetic_column('rare_pair rand_d', data[:, 3])

    def __check_rare_pairs(self, test_id):
        """
        This flags pairs of values, where neither value is by itself rare, but the combination is. This is performed
        on each pair of string columns. The test RARE_COMBINATION covers pairs of numeric columns. For pairs of
        columns with one string and one numeric, there are the VERY_LARGE_GIVEN_VALUE and VERY_SMALL_GIVEN_VALUE tests.
        """
        for col_name_1, col_name_2 in self.__get_binary_column_pairs_unique():
            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if self.orig_df[col_name_1].nunique() > math.sqrt(self.num_rows) or \
               self.orig_df[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue
            vc_1 = self.orig_df[col_name_1].value_counts()
            vc_2 = self.orig_df[col_name_2].value_counts()
            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < math.pow(self.contamination_level, 2):
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < math.pow(self.contamination_level, 2):
                        continue
                    sub_df = self.orig_df[(self.orig_df[col_name_1] == v1) &
                                          (self.orig_df[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue
                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    expected_count = ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows

                    # Check the count is both low and lower than expected.
                    if count_pair < self.contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append((v1, v2))
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                "Rare combinations of values were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )

    def __generate_rare_pair_first_char(self):
        """
        Patterns without exceptions:
        Patterns with exception: The first column starts with characters a, e, i, and the 2nd with b,d,f,h,j,l. a is
            frequently matched with b and d, but in only once with j
        """
        common_vals = [
            ["ax-xxxx", "bx-xxxxxx"],
            ["ax-xxxx", "dx-xxxxxx"],
            ["ex-xxxx", "fx-xxxxxx"],
            ["ex-xxxx", "hx-xxxxxx"],
            ["ix-xxxx", "jx-xxxxxx"],
            ["ix-xxxx", "lx-xxxxxx"]
        ]
        rare_vals = [["ax-xxxx", "jx-xxxxxx"]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self.__add_synthetic_column('rare_pair_first_char all_a', data[:, 0])
        self.__add_synthetic_column('rare_pair_first_char all_b', data[:, 1])

    def __check_rare_pair_first_char(self, test_id):
        # todo: skip where columns are long with multiple words
        # todo: give stats about how frequent each first char is, and how many first chars there are

        first_chars_dict = {}
        for col_name in self.string_cols:
            first_chars_dict[col_name] = pd.Series(self.orig_df[col_name].astype(str).str[:1])

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if first_chars_dict[col_name_1].nunique() > math.sqrt(self.num_rows) or \
                    first_chars_dict[col_name_1].nunique() > math.sqrt(self.num_rows):
                continue
            vc_1 = first_chars_dict[col_name_1].value_counts()
            vc_2 = first_chars_dict[col_name_2].value_counts()
            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < math.pow(self.contamination_level, 2):
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < math.pow(self.contamination_level, 2):
                        continue
                    sub_df = self.orig_df[(first_chars_dict[col_name_1] == v1) &
                                          (first_chars_dict[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue
                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    expected_count = ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows

                    # Check the count is both low and lower than expected.
                    if count_pair < self.contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append((v1, v2))
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                "Rare combinations of first characters were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )

    def __generate_rare_pair_first_word(self):
        """
        Patterns without exceptions: None. This test does not flag patterns, only exceptions
        Patterns with exception: 'rare_pair_first_word all_a' and 'rare_pair_first_word all_b' have 1 rare combination.
        """
        common_vals = [
            ["ax-xxx1", "b1-xxxxx1"],
            ["ax-xxx2", "b1-xxxxx2"],
            ["ax-xxx3", "b2-xxxxx3"],
            ["ax-xxx4", "b2-xxxxx4"],
            ["ex-xxx5", "b3-xxxxx5"],
            ["ex-xxx6", "b3-xxxxx6"],
            ["ix-xxx7", "b4-xxxxx7"],
            ["ix-xxx8", "b4-xxxxx8"],
            ["ix-xxx9", "b4-xxxxx9"],
            ["ix-xxx10", "b6-xxxxx10"]
        ]
        rare_vals = [["ax-xxxx", "b4-xxxxxx"]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self.__add_synthetic_column('rare_pair_first_word all_a', data[:, 0])
        self.__add_synthetic_column('rare_pair_first_word all_b', data[:, 1])

    def __check_rare_pair_first_word(self, test_id):
        first_words_dict = {}
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            first_words_dict[col_name] = pd.Series([x[0] if len(x) >0 else "" for x in col_vals.str.split()])

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            # Skip columns if the first words are as unique as teh values in the column. In this case, the first words
            # have no real meaning on their own.
            if first_words_dict[col_name_1].nunique() >= (self.orig_df[col_name_1].nunique() / 2):
                continue
            if first_words_dict[col_name_2].nunique() >= (self.orig_df[col_name_2].nunique() / 2):
                continue

            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if first_words_dict[col_name_1].nunique() > math.sqrt(self.num_rows) or \
                    first_words_dict[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue
            vc_1 = first_words_dict[col_name_1].value_counts()
            vc_2 = first_words_dict[col_name_2].value_counts()
            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < math.pow(self.contamination_level, 2):
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < math.pow(self.contamination_level, 2):
                        continue
                    sub_df = self.orig_df[(first_words_dict[col_name_1] == v1) &
                                          (first_words_dict[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue
                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    expected_count = ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows

                    # Check the count is both low and lower than expected.
                    if count_pair < self.contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append((v1, v2))
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                "Rare combinations of first words were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )

    def __generate_rare_pair_first_word_val(self):
        """
        Patterns without exceptions: None
        Patterns with exception: The combination of values in 'rare_pair_first_word_val all_a' and
        'rare_pair_first_word_val all_b' are consistently common, with 1 exception
        """
        common_vals = [
            ["ax-xxx1", "A"],
            ["ax-xxx2", "A"],
            ["ax-xxx3", "A"],
            ["ax-xxx4", "A"],
            ["ex-xxx5", "A"],
            ["ex-xxx6", "A"],
            ["ex-xxx7", "B"],
            ["ex-xxx8", "A"],
            ["ix-xxx9", "B"],
            ["ix-xxx10", "A"],
            ["ix-xxx11", "B"],
            ["ix-xxx12", "C"],
            ["ix-xxx13", "D"]
        ]
        rare_vals = [["ax-xxxx", "B"]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self.__add_synthetic_column('rare_pair_first_word_val all_a', data[:, 0])
        self.__add_synthetic_column('rare_pair_first_word_val all_b', data[:, 1])

    def __check_rare_pair_first_word_val(self, test_id):
        first_words_dict = {}
        word_count_dict = {}
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            first_words_dict[col_name] = pd.Series([x[0] if len(x) > 0 else "" for x in col_vals.str.split()])
            word_count_dict[col_name] = pd.Series([len(x) for x in col_vals.str.split()])

        for col_name_1, col_name_2 in self.__get_string_column_pairs():
            # Skip col_name_1 if it's mostly single words
            if statistics.median([x for x in word_count_dict[col_name_1] if x > 0]) <= 1:
                continue

            # Skip col_name_1 if the first words are as unique as the values in the column. In this case, the first
            # words have no real meaning on their own.
            if first_words_dict[col_name_1].nunique() >= (self.orig_df[col_name_1].nunique() / 2):
                continue

            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if first_words_dict[col_name_1].nunique() > math.sqrt(self.num_rows) or \
                    self.orig_df[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue
            vc_1 = first_words_dict[col_name_1].value_counts()
            vc_2 = self.orig_df[col_name_2].value_counts()
            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip empty / Null values
                if v1 == "":
                    continue

                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < math.pow(self.contamination_level, 2):
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < math.pow(self.contamination_level, 2):
                        continue
                    sub_df = self.orig_df[(first_words_dict[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue
                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    expected_count = ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows

                    # Check the count is both low and lower than expected.
                    if count_pair < self.contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append([v1, v2])
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                f"Rare combinations of the first word in {col_name_1} and the value in {col_name_2} were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )

    def __generate_similar_chars(self):
        """
        Patterns without exceptions: The columns "sim_chars rand_a" and "sim_chars all" consistently have a similar
            characters as each other
        Patterns with exception: "sim_chars rand_a" and "sim_chars most" consistently have a similar characters as each
            other, with 1 exception
        """
        self.__add_synthetic_column('sim_chars rand_a',
            [[''.join(random.choice(alphanumeric) for _ in range(10))][0] for i in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim_chars all', self.synth_df['sim_chars rand_a'] + "a")
        self.__add_synthetic_column('sim_chars most', self.synth_df['sim_chars all'])
        self.synth_df.loc[999, 'sim_chars most'] = "abcd"

    def __check_similar_chars(self, test_id):
        """
        This is intended for stings that appers to be ids, codes or other such strings, so this test only covers columns
        that contain exactly one word in each value.
        """
        for col_idx_1 in range(len(self.string_cols)-1):
            col_name_1 = self.string_cols[col_idx_1]
            col_vals = self.orig_df[col_name_1].astype(str).apply(replace_special_with_space)
            word_counts = [0 if x is None else len(x) for x in col_vals.str.split()]
            if max(word_counts) > 1:
                continue
            chars_list_1 = [[""] if x is None else list(x) for x in col_vals]
            for col_idx_2 in range(col_idx_1 + 1, len(self.string_cols)):
                col_name_2 = self.string_cols[col_idx_2]
                col_vals = self.orig_df[col_name_2].apply(replace_special_with_space)
                word_counts = [0 if x is None else len(x) for x in col_vals.str.split()]
                if max(word_counts) > 1:
                    continue
                chars_list_2 = [[""] if x is None else list(x) for x in col_vals]
                test_series = [(len(set(x).union(set(y))) > 0) and
                                   (len(set(x).intersection(set(y))) / len(set(x).union(set(y))) > 0.8)
                               for x, y in zip(chars_list_1, chars_list_2)]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    np.array(test_series),
                    (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar characters '
                     f'as each other')
                )

    def __generate_similar_num_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        num_chars_arr = [np.random.randint(0, 20) for _ in range(self.num_synth_rows)]
        self.__add_synthetic_column('sim_num_chars rand_a',
            [[''.join(random.choice(alphanumeric) for _ in range(num_chars_arr[i]))][0] for i in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim_num_chars all',
            [[''.join(random.choice(alphanumeric) for _ in range(num_chars_arr[i]))][0] for i in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim_num_chars most', self.synth_df['sim_num_chars all'])
        self.synth_df.loc[999, 'sim_num_chars most'] = "abcdefghijklmnopqrstuv"

    def __check_similar_num_chars(self, test_id):
        char_len_dict = {}
        for col_name in self.string_cols:
            char_len_dict[col_name] = self.orig_df[col_name].astype(str).str.len().replace(0, 1)

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            test_series = [0.9 < x/y < 1.1 for x, y in zip(char_len_dict[col_name_1], char_len_dict[col_name_2])]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                np.array(test_series),
                (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar number of characters '
                 f'as each other')
            )

    def __generate_similar_words(self):
        """
        Patterns without exceptions: None
        Patterns with exception: The columns "sim_words all" and "sim_words most" consistently have a similar number
            of words as each other, with exceptions
        """
        self.__add_synthetic_column('sim_words rand_a', [' '.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(3, 6)))
                                                         for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim_words all', [x + " xx" for x in self.synth_df['sim_words rand_a']])
        self.__add_synthetic_column('sim_words most', [x + " xx" for x in self.synth_df['sim_words rand_a']])
        self.synth_df.loc[999, 'sim_words most'] = 'abcdef'

    def __check_similar_words(self, test_id):
        # todo: we can probably put this in process_data(). this is also used in __check_similar_num_words()
        word_counts_dict = {}
        word_counts_medians = {}
        for col_name in self.string_cols:
            word_counts_dict[col_name] = pd.Series([len(x.split()) if (not is_missing(x)) else 0 for x in self.orig_df[col_name]])
            word_counts_not_null = pd.Series([len(x.split()) for x in self.orig_df[col_name] if (not is_missing(x))])
            word_counts_medians[col_name] = word_counts_not_null.median()

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            word_counts_medians_1 = word_counts_medians[col_name_1]
            word_counts_medians_2 = word_counts_medians[col_name_2]
            if word_counts_medians_1 < 3 or word_counts_medians_2 < 3:
                continue
            if word_counts_medians_2 < (word_counts_medians_1 / 2):
                continue
            if word_counts_medians_2 > (word_counts_medians_1 * 2):
                continue

            word_lists_1 = pd.Series([x.split() if (not is_missing(x)) else [] for x in self.orig_df[col_name_1]])
            word_lists_2 = pd.Series([x.split() if (not is_missing(x)) else [] for x in self.orig_df[col_name_2]])
            similarity_series = [0 if ((len(a) == 0) or (len(b) == 0)) else float(len(set(a).intersection(set(b)))) / len(set(a).union(set(b)))
                                 for a, b in zip(word_lists_1, word_lists_2)]
            for test_threshold in np.arange(1.0, 0.6, -0.1):
                test_series = [x >= test_threshold for x in similarity_series]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                if test_series.tolist().count(False) < self.contamination_level:
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_1, col_name_2]),
                        [col_name_1, col_name_2],
                        np.array(test_series),
                        (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar number of words '
                         f'as each other')
                    )
                    break

    def __generate_similar_num_words(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('sim_num_words rand_a', [' '.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(3,6)))
                                                          for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim_num_words all', [' '.join(np.random.choice(list(string.ascii_lowercase), 5))
                                                             for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('sim_num_words most', [' '.join(np.random.choice(list(string.ascii_lowercase), 5))
                                                          for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'sim_num_words most'] = 'abcdef'

    def __check_similar_num_words(self, test_id):
        # todo: handle where one column or the other has None or Nan values but otherwise similar
        word_counts_dict = {}
        word_counts_medians = {}
        for col_name in self.string_cols:
            word_counts_dict[col_name] = pd.Series([len(x.split()) if (not is_missing(x)) else 0 for x in self.orig_df[col_name]])
            word_counts_not_null = pd.Series([len(x.split()) for x in self.orig_df[col_name] if (not is_missing(x))])
            word_counts_medians[col_name] = word_counts_not_null.median()

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            word_counts_medians_1 = word_counts_medians[col_name_1]
            word_counts_medians_2 = word_counts_medians[col_name_2]
            if word_counts_medians_1 < 3 or word_counts_medians_2 < 3:
                continue
            if word_counts_medians_2 < (word_counts_medians_1 / 2):
                continue
            if word_counts_medians_2 > (word_counts_medians_1 * 2):
                continue

            word_counts_1 = word_counts_dict[col_name_1]
            word_counts_2 = word_counts_dict[col_name_2]
            if len(word_counts_1.value_counts()) == 1 and len(word_counts_2.value_counts()) == 1:
                continue
            similarity_series = [abs(x - y) for x, y in zip(word_counts_1, word_counts_2)]
            for test_threshold in range(3, -1, -1):
                test_series = [x <= test_threshold for x in similarity_series]
                test_series = (test_series | self.orig_df[col_name_1].isnull()) | self.orig_df[col_name_2].isnull()
                if test_series.tolist().count(False) < self.contamination_level:
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_1, col_name_2]),
                        [col_name_1, col_name_2],
                        np.array(test_series),
                        (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar number of words '
                         f'as each other')
                    )
                    break

    def __generate_same_first_chars(self):
        """
        Patterns without exceptions: "same_start rand_a" and "same_start all" consistently share the same first 9
            characters
        Patterns with exception:
        """
        self.__add_synthetic_column('same_start rand_a', [''.join(np.random.choice(list(string.ascii_lowercase), 10))
                                                          for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same_start all', [x + "xxxx" for x in self.synth_df['same_start rand_a']])
        self.__add_synthetic_column('same_start most', [x + "xxxx" for x in self.synth_df['same_start rand_a']])
        self.synth_df.loc[999, 'same_start most'] = 'ABCDE'

    def __check_same_first_chars(self, test_id):
        def check_column(col_name):
            first_chars_series = self.orig_df[col_name].astype(str).str.slice(0, 1)  # Get the first letter of each value
            counts_series = first_chars_series.value_counts(normalize=True)  # Get the counts for each first letter
            if len(counts_series) < 10:
                return False
            if counts_series[0] > 50.0:
                return False
            return True

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            # Check if either column always starts with the same characters anyway.
            if not check_column(col_name_1) or not check_column(col_name_2):
                continue

            # Determine how many characters we can potentially check for matches between the columns
            avg_len_1 = pd.Series([len(x.split()) for x in self.orig_df[col_name_1].astype(str) if (not is_missing(x))]).mean()
            avg_len_2 = pd.Series([len(x.split()) for x in self.orig_df[col_name_2].astype(str) if (not is_missing(x))]).mean()
            max_check = min(avg_len_1, avg_len_2)

            # Determine the length of longest substrings in the values in both columns that match, such that there
            # are sufficient matches to consider this a valid pattern (less than contamination_level exceptions)
            max_number_matching = -1
            # todo: handle where one or the other of the columns has Null values
            for num_chars_checking in range(1, int(max_check) + 1):
                first_chars_series_1 = self.orig_df[col_name_1].fillna("").astype(str).str.slice(0, num_chars_checking)
                first_chars_series_2 = self.orig_df[col_name_2].fillna("").astype(str).str.slice(0, num_chars_checking)
                num_matching = len([x == y for x, y in zip(first_chars_series_1, first_chars_series_2)])
                if num_matching < (self.num_rows - self.contamination_level):
                    break
                max_number_matching = num_chars_checking

            if max_number_matching <= 0:
                continue

            first_chars_series_1 = self.orig_df[col_name_1].astype(str).str.slice(0, max_number_matching)
            first_chars_series_2 = self.orig_df[col_name_2].astype(str).str.slice(0, max_number_matching)
            test_series = np.array([x == y for x, y in zip(first_chars_series_1, first_chars_series_2)])
            test_series = (test_series | self.orig_df[col_name_1].isnull()) | self.orig_df[col_name_2].isnull()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                test_series,
                (f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same first {max_number_matching} '
                 f'characters'))

    def __generate_same_first_word(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        # Test case where words separated by white space
        self.__add_synthetic_column('same_start_word rand_a', [''.join(np.random.choice(['a', 'b', ' '], 10))
                                                          for _ in range(self.num_synth_rows)])
        self.synth_df['same_start_word rand_a'] += " ab"  # Ensure no values are entirely blank
        self.__add_synthetic_column('same_start_word all_a', [x[0] + " wxyz" for x in self.synth_df['same_start_word rand_a'].str.split()])
        self.__add_synthetic_column('same_start_word most_a', self.synth_df['same_start_word all_a'])
        self.synth_df.loc[999, 'same_start_word most_a'] = 'ABCDE'

        # Test case where words separated by hyphens
        self.__add_synthetic_column('same_start_word rand_b', [''.join(np.random.choice(['a', 'b', '-'], 10))
                                                               for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same_start_word all_b', [x[0] + " wxyz" for x in  self.synth_df['same_start_word rand_b'].str.split()])
        self.__add_synthetic_column('same_start_word most_b', self.synth_df['same_start_word all_b'])
        self.synth_df.loc[999, 'same_start_word most_b'] = 'ABCDE'

    def __check_same_first_word(self, test_id):

        # todo: probably need to call replace_special_with_space() as in __check_first_word()
        # Get the first word in each string
        sample_first_words_dict = {}
        first_words_dict = {}
        for col_name in self.string_cols:
            sample_first_words_dict[col_name] = [x[0] if (x and (len(x) > 0)) else "" for x in self.sample_df[col_name].astype(str).str.split()]
            first_words_dict[col_name] = [x[0] if (x and (len(x) > 0)) else "" for x in self.orig_df[col_name].astype(str).str.split()]

        for col_name_1, col_name_2 in self.__get_string_column_pairs():
            test_series = [x == y for x, y in zip(sample_first_words_dict[col_name_1], sample_first_words_dict[col_name_2])]
            if test_series.count(False) > 1:
                continue
            test_series = [x == y for x, y in zip(first_words_dict[col_name_1], first_words_dict[col_name_2])]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                np.array(test_series),
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same first word'
            )

    def __generate_same_last_word(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        # Test case where words separated by white space
        self.__add_synthetic_column('same_last_word rand_a', [''.join(np.random.choice(['a', 'b', ' '], 10))
                                                               for _ in range(self.num_synth_rows)])
        self.synth_df['same_last_word rand_a'] = "ab " +  self.synth_df['same_last_word rand_a'] # Ensure no values are entirely blank
        self.__add_synthetic_column('same_last_word all_a', ["wxyz " + x[-1] for x in  self.synth_df['same_last_word rand_a'].astype(str).str.split()])
        self.__add_synthetic_column('same_last_word most_a', self.synth_df['same_last_word all_a'])
        self.synth_df.loc[999, 'same_last_word most_a'] = 'ABCDE'

        # Test case where words separated by hyphens
        self.__add_synthetic_column('same_last_word rand_b', [''.join(np.random.choice(['a', 'b', '-'], 10))
                                                               for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same_last_word all_b', ["wxyz " + x[-1] for x in  self.synth_df['same_last_word rand_b'].astype(str).str.split()])
        self.__add_synthetic_column('same_last_word most_b', self.synth_df['same_last_word all_b'])
        self.synth_df.loc[999, 'same_last_word most_b'] = 'ABCDE'

    def __check_same_last_word(self, test_id):

        # Get the last word in each string
        sample_last_words_dict = {}
        last_words_dict = {}
        for col_name in self.string_cols:
            sample_last_words_dict[col_name] = [None if is_missing(x) else x[0] for x in self.sample_df[col_name].astype(str).str.split()]
            last_words_dict[col_name] = [None if is_missing(x) else x[0] for x in self.orig_df[col_name].astype(str).str.split()]

        for col_name_1, col_name_2 in self.__get_string_column_pairs():
            test_series = [x == y for x, y in zip(sample_last_words_dict[col_name_1], sample_last_words_dict[col_name_2])]
            if test_series.count(False) > 1:
                continue
            test_series = [x == y for x, y in zip(last_words_dict[col_name_1], last_words_dict[col_name_2])]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                np.array(test_series),
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same last word'
            )

    def __generate_same_alpha_chars(self):
        """
        Patterns without exceptions: None
        Patterns with exception: "same_alpha all" and "same_alpha most" consistently share the same alphabetic
            characters, with 1 exception
        """
        self.__add_synthetic_column('same_alpha rand_a', [''.join(np.random.choice(list(string.ascii_lowercase), 10))
                                                          for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same_alpha all', self.synth_df['same_alpha rand_a'].str[:1] +
                                    ''.join(np.random.choice(list(string.digits), 10)))
        self.__add_synthetic_column('same_alpha most', self.synth_df['same_alpha all'])
        self.synth_df.loc[999, 'same_alpha most'] = 'ABCDE'

    def __check_same_alpha_chars(self, test_id):
        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            # Test on a sample.
            alpha_col_1 = self.sample_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_1 = [[c for c in x if c and c.isalpha()] for x in alpha_col_1]
            if [len(x) > 0 for x in alpha_col_1].count(False) > 1:
                continue
            alpha_col_2 = self.sample_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_2 = [[c for c in x if c and c.isalpha()] for x in alpha_col_2]
            if [len(x) > 0 for x in alpha_col_2].count(False) > 1:
                continue
            test_series = [set(x) == set(y) for x, y in zip(alpha_col_1, alpha_col_2)]
            if test_series.count(False) > 1:
                continue

            # Test on the actual data
            alpha_col_1 = self.orig_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_1 = [[c for c in x if c and c.isalpha()] for x in alpha_col_1]
            alpha_col_2 = self.orig_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_2 = [[c for c in x if c and c.isalpha()] for x in alpha_col_2]
            test_series = [set(x) == set(y) for x, y in zip(alpha_col_1, alpha_col_2)]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                np.array(test_series),
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same alphabetic characters '
            )

    def __generate_same_numeric_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('same_num rand_a', [''.join(np.random.choice(list(string.digits), 10))
                                                          for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same_num all', self.synth_df['same_num rand_a'].str[:1] +
                                    ''.join(np.random.choice(list(string.ascii_lowercase), 10)))
        self.__add_synthetic_column('same_num most', self.synth_df['same_num all'])
        self.synth_df.loc[999, 'same_num most'] = '1234'

    def __check_same_numeric_chars(self, test_id):
        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            # Test on a sample
            digits_col_1 = self.sample_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            digits_col_1 = [[c for c in x if c and c.isdigit()] for x in digits_col_1]
            if [len(x) > 0 for x in digits_col_1].count(False) > 1:
                continue
            digits_col_2 = self.sample_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            digits_col_2 = [[c for c in x if c and c.isdigit()] for x in digits_col_2]
            if [len(x) > 0 for x in digits_col_2].count(False) > 1:
                continue
            test_series = [set(x) == set(y) for x, y in zip(digits_col_1, digits_col_2)]
            if test_series.count(False) > 1:
                continue

            # Test on the full columns
            digits_col_1 = self.orig_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            digits_col_1 = [[c for c in x if c and c.isdigit()] for x in digits_col_1]
            digits_col_2 = self.orig_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            digits_col_2 = [[c for c in x if c and c.isdigit()] for x in digits_col_2]
            test_series = [set(x) == set(y) for x, y in zip(digits_col_1, digits_col_2)]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                np.array(test_series),
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same numeric characters '
            )

    def __generate_same_special_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('same_special rand_a', [''.join(np.random.choice(['$', '%', '#', '@', '!'], 10))
                                                        for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('same_special all', self.synth_df['same_special rand_a'].str[:1] +
                                    ''.join(np.random.choice(list(string.ascii_lowercase), 10)))
        self.__add_synthetic_column('same_special most', self.synth_df['same_special all'])
        self.synth_df.loc[999, 'same_special most'] = '&&^%'

    def __check_same_special_chars(self, test_id):
        # todo: in display, add a column listing the special chars
        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            # Test on a sample.
            special_col_1 = self.sample_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            special_col_1 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()))] for x in special_col_1]
            if [len(x) > 0 for x in special_col_1].count(False) > 1:
                continue
            special_col_2 = self.sample_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            special_col_2 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()))] for x in special_col_2]
            if [len(x) > 0 for x in special_col_2].count(False) > 1:
                continue
            test_series = [set(x) == set(y) for x, y in zip(special_col_1, special_col_2)]
            if test_series.count(False) > 1:
                continue

            # Test on the full columns
            special_col_1 = self.orig_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            special_col_1 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()))] for x in special_col_1]
            if [len(x) > 0 for x in special_col_1].count(False) > self.contamination_level:
                continue
            special_col_2 = self.orig_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            special_col_2 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()))] for x in special_col_2]
            if [len(x) > 0 for x in special_col_2].count(False) > self.contamination_level:
                continue
            test_series = [set(x) == set(y) for x, y in zip(special_col_1, special_col_2)]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                np.array(test_series),
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same special characters'
            )

    def __generate_a_prefix_of_b(self):
        """
        Patterns without exceptions: "a_prefix_b rand_a" is consistently the same as the first characters of
            "a_prefix_b all"
        Patterns with exception: "a_prefix_b rand_a" is consistently the same as the first characters of
            "a_prefix_b most", with exceptions
        """
        self.__add_synthetic_column('a_prefix_b rand_a', [''.join(np.random.choice(list(string.ascii_lowercase), 10))
                                                            for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('a_prefix_b all', self.synth_df['a_prefix_b rand_a'] +
                                    ''.join(np.random.choice(list(string.ascii_lowercase), 10)))
        self.__add_synthetic_column('a_prefix_b most', self.synth_df['a_prefix_b all'])
        self.synth_df.loc[999, 'a_prefix_b most'] = 'abcdef'

    def __check_a_prefix_of_b(self, test_id):
        for col_name_1, col_name_2 in self.__get_string_column_pairs():
            # todo: test on a sample first
            test_series = [True if (is_missing(x) or is_missing(y)) else ((len(x) < len(y)) and (x == y[:len(x)]))
                           for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self.__process_analysis_binary(
                test_id,
                self.get_col_set_name([col_name_1, col_name_2]),
                [col_name_1, col_name_2],
                np.array(test_series),
                f'Columns "{col_name_1}" is consistently the same as the first characters of "{col_name_2}"'
            )

    def __generate_correlated_alpha(self):
        """
        Patterns without exceptions: "correlated_alpha rand_a" is consistently similar, with regards to percentile, to
            "correlated_alpha rand_b"
        Patterns with exception: "correlated_alpha rand_a" is consistently similar, with regards to percentile, to
            "correlated_alpha most", with exceptions, and:
            "correlated_alpha rand_b" is consistently similar, with regards to percentile, to
            "correlated_alpha most", with exceptions
        """
        list_a = sorted([''.join(np.random.choice(list(string.ascii_lowercase), 10))
                           for _ in range(self.num_synth_rows)])
        list_b = sorted([''.join(np.random.choice(list(string.ascii_lowercase), 10))
                          for _ in range(self.num_synth_rows)])
        c = list(zip(list_a, list_b))
        random.shuffle(c)
        list_a, list_b = zip(*c)

        self.__add_synthetic_column('correlated_alpha rand_a', list_a)
        self.__add_synthetic_column('correlated_alpha rand_b', list_b)
        list_b = list(list_b)
        list_b[-1] = 'aaaaaaaaaa'
        self.__add_synthetic_column('correlated_alpha most', list_b)

    def __check_correlated_alpha(self, test_id):
        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique():
            if self.orig_df[col_name_1].nunique(dropna=True) == 1:
                continue
            if self.orig_df[col_name_2].nunique(dropna=True) == 1:
                continue

            spearancorr = abs(self.orig_df[col_name_1].corr(self.orig_df[col_name_2], method='spearman'))
            if spearancorr >= 0.95:
                col_1_percentiles = self.orig_df[col_name_1].rank(pct=True)
                col_2_percentiles = self.orig_df[col_name_2].rank(pct=True)

                # Test for positive correlation
                test_series = np.array([abs(x-y) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_1}" is consistently similar, with regards to percentile, to "{col_name_2}"')

                # Test for negative correlation
                test_series = np.array([abs(x-(1.0 - y)) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self.__process_analysis_binary(
                    test_id,
                    f'"{col_name_1}" AND "{col_name_2}"',
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_1}" is consistently inversely similar in percentile to "{col_name_2}"')

    ####################################################################################
    # Data consistency checks for one non-numeric column and one numeric column
    ####################################################################################

    def __generate_very_large_given(self):
        """
        Patterns without exceptions:
        Patterns with exception: 'large_given most' contains one row with value 290, which is common for the column
            but not common when 'large_given rand' is 'C'
        """
        self.__add_synthetic_column('large_given rand', ['A']*100 + ["B"]*100 + ['C']*(self.num_synth_rows - 200))
        self.__add_synthetic_column('large_given all',np.concatenate([
            np.random.randint(200, 300, 100),
            np.random.randint(100, 200, 100),
            np.random.randint(0, 100, (self.num_synth_rows - 200))
        ]))
        self.__add_synthetic_column('large_given most', self.synth_df['large_given all'])
        self.synth_df.loc[999, 'large_given most'] = 290

    def __check_very_large_given(self, test_id):
        for col_name_1 in self.string_cols + self.binary_cols:
            if self.orig_df[col_name_1].nunique() > 10:
                continue
            vc = self.orig_df[col_name_1].value_counts()
            common_values = []
            for v in vc.index:
                if vc[v] > math.sqrt(self.num_rows):
                    common_values.append(v)
            for col_name_2 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                for v in common_values:
                    sub_df = self.orig_df[[col_name_2]][self.orig_df[col_name_1] == v]
                    q1 = sub_df[col_name_2].quantile(0.25)
                    q3 = sub_df[col_name_2].quantile(0.75)
                    upper_limit = q3 + (self.iqr_limit * (q3 - q1))
                    res = sub_df[col_name_2] <= upper_limit
                    if 0 < res.tolist().count(False) <= self.contamination_level:
                        index_of_large = [x for x, y in zip(sub_df.index, res) if y == False]
                        for i in index_of_large:
                            test_series[i] = False
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    np.array(test_series),
                    f'"{col_name_2}" contains very large values given the value in "{col_name_1}"',
                    allow_patterns=False  # todo: ones that set allow_patterns=False may get ,. in string. check for that. maybe just do a replace()
                )

    def __generate_very_small_given(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('small_given rand', ['A']*100 + ["B"]*100 + ['C']*(self.num_synth_rows - 200))
        self.__add_synthetic_column('small_given all',np.concatenate([
            np.random.randint(0, 100, 100),
            np.random.randint(100, 200, 100),
            np.random.randint(200, 300, (self.num_synth_rows - 200))
        ]))
        self.__add_synthetic_column('small_given most', self.synth_df['small_given all'])
        self.synth_df.loc[999, 'small_given most'] = 5

    def __check_very_small_given(self, test_id):
        for col_name_1 in self.string_cols + self.binary_cols:
            if self.orig_df[col_name_1].nunique() > 10:
                continue
            vc = self.orig_df[col_name_1].value_counts()
            common_values = []
            for v in vc.index:
                if vc[v] > math.sqrt(self.num_rows):
                    common_values.append(v)
            for col_name_2 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                for v in common_values:
                    sub_df = self.orig_df[[col_name_2]][self.orig_df[col_name_1] == v]
                    q1 = sub_df[col_name_2].quantile(0.25)
                    q3 = sub_df[col_name_2].quantile(0.75)
                    lower_limit = q1 - (self.iqr_limit * (q3 - q1))
                    res = ((sub_df[col_name_2] >= lower_limit) | (sub_df[col_name_2].isna()))
                    if 0 < res.tolist().count(False) <= self.contamination_level:
                        index_of_small = [x for x, y in zip(sub_df.index, res) if not y]
                        for i in index_of_small:
                            test_series[i] = False
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_2}" contains very small values given the value in "{col_name_1}"',
                    allow_patterns=False
                )

    def __generate_large_given_prefix(self):
        """
        Patterns without exceptions:
        Patterns with exception: 'large_given most' contains one row with value 290, which is common for the column
            but not common when 'large_given rand' is 'C'
        """
        self.__add_synthetic_column('large_given_prefix rand',
                                    ['A-' + np.random.choice(list(string.ascii_letters)) for _ in range(100)] + \
                                    ['B-' + np.random.choice(list(string.ascii_letters)) for _ in range(100)] + \
                                    ['C-' + np.random.choice(list(string.ascii_letters)) for _ in range(self.num_synth_rows - 200)])
        self.__add_synthetic_column('large_given_prefix all', np.concatenate([
            np.random.randint(200, 300, 100),
            np.random.randint(100, 200, 100),
            np.random.randint(0, 100, (self.num_synth_rows - 200))
        ]))
        self.__add_synthetic_column('large_given_prefix most', self.synth_df['large_given_prefix all'])
        self.synth_df.loc[999, 'large_given_prefix most'] = 290

    def __check_large_given_prefix(self, test_id):
        # todo: when show non-flagged, show with the flagged prefixes
        # todo: dont flag Null values
        for col_name_1 in self.string_cols:
            col_vals = self.orig_df[col_name_1].astype(str).apply(replace_special_with_space)

            # Check the column's values are usually more than 1 word
            word_arr = col_vals.str.split()
            word_counts_arr = pd.Series([len(x) for x in word_arr])
            if word_counts_arr.quantile(0.75) <= 1:
                continue

            first_words = pd.Series([x[0] if len(x) > 0 else "" for x in col_vals.str.split()])
            if first_words.nunique() > 10:
                continue
            vc = first_words.value_counts()
            common_values = []
            for v in vc.index:
                if vc[v] > math.sqrt(self.num_rows):
                    common_values.append(v)
            for col_name_2 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                flagged_prefixes = []
                for v in common_values:
                    sub_df = self.orig_df[[col_name_2]][first_words == v]
                    q1 = sub_df[col_name_2].quantile(0.25)
                    q3 = sub_df[col_name_2].quantile(0.75)
                    upper_limit = q3 + (self.iqr_limit * (q3 - q1))
                    res = sub_df[col_name_2] <= upper_limit
                    if 0 < res.tolist().count(False) <= self.contamination_level:
                        flagged_prefixes.append(v)
                        index_of_large = [x for x, y in zip(sub_df.index, res) if not y]
                        for i in index_of_large:
                            test_series[i] = False
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    np.array(test_series),
                    (f'"{col_name_2}" contains very large values given the first word/prefix {flagged_prefixes} in '
                     f'"{col_name_1}"'),
                    allow_patterns=False
                )

    def __generate_small_given_prefix(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('small_given_prefix rand',
                                    ['A-' + np.random.choice(list(string.ascii_letters))]*100 +
                                    ['B-' + np.random.choice(list(string.ascii_letters))]*100 +
                                    ['C-' + np.random.choice(list(string.ascii_letters))]*(self.num_synth_rows - 200))
        self.__add_synthetic_column('small_given_prefix all', np.concatenate([
            np.random.randint(0, 100, 100),
            np.random.randint(100, 200, 100),
            np.random.randint(200, 300, (self.num_synth_rows - 200))
        ]))
        self.__add_synthetic_column('small_given_prefix most', self.synth_df['small_given_prefix all'])
        self.synth_df.loc[999, 'small_given_prefix most'] = 8

    def __check_small_given_prefix(self, test_id):
        # todo: when show non-flagged, show with the flagged prefixes
        # todo: dont flag Null values
        for col_name_1 in self.string_cols:
            col_vals = self.orig_df[col_name_1].astype(str).apply(replace_special_with_space)

            # Check the column's values are usually more than 1 word
            word_arr = col_vals.str.split()
            word_counts_arr = pd.Series([len(x) for x in word_arr])
            if word_counts_arr.quantile(0.75) <= 1:
                continue

            first_words = pd.Series([x[0] if len(x) > 0 else "" for x in col_vals.str.split()])
            if first_words.nunique() > 10:
                continue
            vc = first_words.value_counts()
            common_values = []
            for v in vc.index:
                if vc[v] > math.sqrt(self.num_rows):
                    common_values.append(v)
            for col_name_2 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                flagged_prefixes = []
                for v in common_values:
                    sub_df = self.orig_df[[col_name_2]][first_words == v]
                    q1 = sub_df[col_name_2].quantile(0.25)
                    q3 = sub_df[col_name_2].quantile(0.75)
                    lower_limit = q1 - (self.iqr_limit * (q3 - q1))
                    res = sub_df[col_name_2] > lower_limit
                    if 0 < res.tolist().count(False) <= self.contamination_level:
                        flagged_prefixes.append(v)
                        index_of_large = [x for x, y in zip(sub_df.index, res) if not y]
                        for i in index_of_large:
                            test_series[i] = False
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2]),
                    [col_name_1, col_name_2],
                    np.array(test_series),
                    (f'"{col_name_2}" contains very small values given the first word/prefix {flagged_prefixes} in '
                     f'"{col_name_1}"'),
                    allow_patterns=False
                )

    def __generate_grouped_strings_by_numeric(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        group_a_len = group_b_len = self.num_synth_rows // 3
        group_c_len = self.num_synth_rows - (group_a_len + group_b_len)
        df = pd.DataFrame({
            'num_grp_str rand_i': sorted(np.random.random(self.num_synth_rows)),
            'num_grp_str rand_s': np.random.choice(['a', 'b', 'c'], self.num_synth_rows),
            'num_grp_str all': ['a'] * group_a_len + ['b'] * group_b_len + ['c'] * group_c_len,
            'num_grp_str most': ['a'] * group_a_len + ['b'] * group_b_len + ['c'] * (group_c_len - 1) + ['a']
        })
        df = df.sample(n=len(df))  # shuffle the dataframe
        df = df.reset_index(drop=True)
        for col_name in df.columns:
            self.__add_synthetic_column(col_name, df[col_name])

    def __check_grouped_strings_by_numeric(self, test_id):
        """
        This is similar to GROUPED_STRINGS, but checks where the string/binary values are grouped when sorted by a
        numeric or date column, as opposed to the given order of the table.
        """
        for col_name_1 in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name_1].nunique() < (self.num_valid_rows[col_name_1] - self.contamination_level):
                continue
            for col_name_2 in self.string_cols + self.binary_cols:
                if self.orig_df[col_name_2].nunique() > math.sqrt(self.num_valid_rows[col_name_2]):
                    continue
                df = self.orig_df.copy()
                df = df.sort_values(col_name_1)
                col_series = df[col_name_2]
                self.__check_grouped_stings_column(test_id, col_name_2, col_series)

    ####################################################################################
    # Data consistency checks for two string and one numeric column
    ####################################################################################

    def __generate_large_given_pair(self):
        """
        Patterns without exceptions:
        Patterns with exception: Row 999 contains a rare combination of values, so it is not possible to check for
            rare numeric values given these. Row 998 contains a value in 'large_given_pair most' which is common, but
            not for 'd', 'b' in the two string columns.
        """
        common_vals = [
            ['a', 'a'],
            ['a', 'b'],
            ['b', 'a'],
            ['a', 'c'],
            ['d', 'b']]
        rare_vals = [['b', 'c']]

        data = np.array([x*(self.num_synth_rows // len(common_vals)) for x in common_vals]).reshape(-1, 2)
        data = np.vstack([data[:self.num_synth_rows-1], rare_vals])
        self.__add_synthetic_column('large_given_pair rand_a', data[:, 0])
        self.__add_synthetic_column('large_given_pair rand_b', data[:, 1])
        self.__add_synthetic_column('large_given_pair all', np.concatenate([
                                        np.random.randint(200, 300, 100),
                                        np.random.randint(100, 200, 100),
                                        np.random.randint(0, 100, (self.num_synth_rows - 200))
                                    ]))
        self.__add_synthetic_column('large_given_pair most', self.synth_df['large_given_pair all'])
        self.synth_df.loc[998, 'large_given_pair most'] = 290

    def __check_large_given_pair(self, test_id):
        """
        With many Null values, the count of each pair of values in the 2 string columns may be too low to execute this
        test.
        """

        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique(): # todo: we should check the binary columns as well
            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if self.orig_df[col_name_1].nunique() > math.sqrt(self.num_rows) or \
                    self.orig_df[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue

            # Identify the common values in the two binary/categorical columns
            vc_1 = self.orig_df[col_name_1].value_counts()
            common_values_1 = []
            for v in vc_1.index:
                if vc_1[v] > math.sqrt(self.num_rows):
                    common_values_1.append(v)

            vc_2 = self.orig_df[col_name_2].value_counts()
            common_values_2 = []
            for v in vc_2.index:
                if vc_2[v] > math.sqrt(self.num_rows):
                    common_values_2.append(v)

            for col_name_3 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                for v1 in common_values_1:
                    for v2 in common_values_2:
                        sub_df = self.orig_df[(self.orig_df[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]
                        if len(sub_df) < 100:
                            continue

                        q1 = sub_df[col_name_3].quantile(0.25)
                        q3 = sub_df[col_name_3].quantile(0.75)
                        upper_limit = q3 + (self.iqr_limit * (q3 - q1))
                        res = sub_df[col_name_3] <= upper_limit
                        if 0 < res.tolist().count(False) <= self.contamination_level:
                            index_of_large = [x for x, y in zip(sub_df.index, res) if not y]
                            for i in index_of_large:
                                test_series[i] = False
                        test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_3].isna()
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_1, col_name_2, col_name_3]),
                    [col_name_1, col_name_2, col_name_3],
                    np.array(test_series),
                    (f'"{col_name_3}" contains very large values given the values in "{col_name_1}" and '
                     f'"{col_name_2}"'),
                    allow_patterns=False
                )

    def __generate_small_given_pair(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        """
        Patterns without exceptions:
        Patterns with exception: Row 999 contains a rare combination of values, so it is not possible to check for
            rare numeric values given these. Row 998 contains a value in 'large_given_pair most' which is common, but
            not for 'd', 'b' in the two string columns.
        """
        common_vals = [
            ['a', 'a'],
            ['a', 'b'],
            ['b', 'a'],
            ['a', 'c'],
            ['d', 'b']]
        rare_vals = [['b', 'c']]

        data = np.array([x*(self.num_synth_rows // len(common_vals)) for x in common_vals]).reshape(-1, 2)
        data = np.vstack([data[:self.num_synth_rows-1], rare_vals])
        self.__add_synthetic_column('small_given_pair rand_a', data[:, 0])
        self.__add_synthetic_column('small_given_pair rand_b', data[:, 1])
        self.__add_synthetic_column('small_given_pair all', np.concatenate([
            np.random.randint(0, 100, 100),
            np.random.randint(100, 200, 100),
            np.random.randint(200, 300, (self.num_synth_rows - 200))
        ]))
        self.__add_synthetic_column('small_given_pair most', self.synth_df['small_given_pair all'])
        self.synth_df.loc[998, 'small_given_pair most'] = 4

    def __check_small_given_pair(self, test_id):
        for col_name_1, col_name_2 in self.__get_string_column_pairs_unique(): # todo: we should check the binary columns as well
            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if self.orig_df[col_name_1].nunique() > math.sqrt(self.num_rows) or \
                    self.orig_df[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue

            # Identify the common values in the two binary/categorical columns
            vc_1 = self.orig_df[col_name_1].value_counts()
            common_values_1 = []
            for v in vc_1.index:
                if vc_1[v] > math.sqrt(self.num_rows):
                    common_values_1.append(v)

            vc_2 = self.orig_df[col_name_2].value_counts()
            common_values_2 = []
            for v in vc_2.index:
                if vc_2[v] > math.sqrt(self.num_rows):
                    common_values_2.append(v)

            for col_name_3 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                for v1 in common_values_1:
                    for v2 in common_values_2:
                        sub_df = self.orig_df[(self.orig_df[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]
                        if len(sub_df) < 100:
                            continue

                        q1 = sub_df[col_name_3].quantile(0.25)
                        q3 = sub_df[col_name_3].quantile(0.75)
                        lower_limit = q1 - (self.iqr_limit * (q3 - q1))
                        res = sub_df[col_name_3] > lower_limit
                        if 0 < res.tolist().count(False) <= self.contamination_level:
                            index_of_large = [x for x, y in zip(sub_df.index, res) if not y]
                            for i in index_of_large:
                                test_series[i] = False
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name([col_name_1, col_name_2, col_name_3]),
                            [col_name_1, col_name_2, col_name_3],
                            np.array(test_series),
                            (f'"{col_name_3}" contains very small values given the values in "{col_name_1}" and '
                             f'"{col_name_2}"'),
                            allow_patterns=False
                        )

    ####################################################################################
    # Data consistency checks for one string/binary column and two numeric
    ####################################################################################

    def __generate_corr_given_val(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        list_a = sorted([random.randint(1, 1_000) for _ in range(500)])
        list_b = sorted([random.randint(1, 2_000) for _ in range(500)])
        c = list(zip(list_a, list_b))
        random.shuffle(c)
        list_a, list_b = zip(*c)

        list_c = sorted([random.randint(1, 1_000) for _ in range(self.num_synth_rows - 500)])
        list_d = sorted([random.randint(1, 2_000) for _ in range(self.num_synth_rows - 500)], reverse=True)
        c = list(zip(list_c, list_d))
        random.shuffle(c)
        list_c, list_d = zip(*c)

        self.__add_synthetic_column('corr_given_val rand_string', ['A']*500 + ['B']*(self.num_synth_rows - 500))
        self.__add_synthetic_column('corr_given_val rand_a', list_a + list_c)
        self.__add_synthetic_column('corr_given_val rand_b', list_b + list_d)
        list_b = list(list_b)
        list_b[-1] = list_b[-1] / 10.0
        self.__add_synthetic_column('corr_given_val most', list(list_b) + list(list_d))

    def __check_corr_given_val(self, test_id):

        # From: https://stackoverflow.com/questions/71844846/is-there-a-faster-way-to-get-correlation-coefficents
        def pairwise_correlation(A, B):
            am = A - np.mean(A, axis=0)
            bm = B - np.mean(B, axis=0)
            return am.T @ bm / (np.sqrt(
                np.sum(am**2, axis=0)).T * np.sqrt(
                np.sum(bm**2, axis=0)))

        # Create a numpy array of just the numeric columns for efficiency
        numeric_np = self.orig_df[self.numeric_cols].values

        for col_idx, col_cond in enumerate(self.string_cols + self.binary_cols):
            if self.verbose >= 2:
                print(f" Examining column {col_idx} of {len(self.string_cols + self.binary_cols)} string and binary columns")
            conditioning_vals = self.orig_df[col_cond].unique()
            if len(conditioning_vals) > 10:
                continue

            num_pairs, numeric_pairs = self.__get_numeric_column_pairs_unique()
            for num_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs):
                if self.verbose >= 2 and num_idx > 0 and num_idx % 10_000 == 0:
                    print(f"  Examining column set {num_idx:,} of {num_pairs:,} pairs of numeric columns.")

                if self.orig_df[col_name_1].nunique(dropna=True) == 1:
                    continue
                if self.orig_df[col_name_2].nunique(dropna=True) == 1:
                    continue

                # Get the indexes of the 2 numeric columns
                col_idx_1 = self.numeric_cols.index(col_name_1)
                col_idx_2 = self.numeric_cols.index(col_name_2)

                # Skip any pairs of columns that are correlated even if not conditioning on another column
                corr = pairwise_correlation(self.sample_df[col_name_1], self.sample_df[col_name_2])
                if abs(corr) >= 0.995:
                    continue

                all_subsets_correlated = True
                test_series = [True] * self.num_rows
                for val in conditioning_vals:
                    if val is None:
                        idxs = np.where(self.orig_df[col_cond].isna())
                    else:
                        idxs = np.where(self.orig_df[col_cond] == val)
                    sub_np = numeric_np[idxs]
                    if len(sub_np) < 100:
                        continue

                    # First test on a sample of rows
                    sample_sub_df = sub_np[:50]
                    try:
                        corr = pairwise_correlation(sub_np[col_idx_1], sub_np[col_idx_2])
                    except:
                        p= 9
                    if abs(corr) < 0.995:
                        all_subsets_correlated = False
                        break

                    # Second, test with the full columns
                    corr = pairwise_correlation(numeric_np[col_idx_1], numeric_np[col_idx_2])
                    if abs(corr) < 0.995:
                        all_subsets_correlated = False
                        break

                    # pairwise_correlation() is fast, but does not distinguish positive from negative correlation
                    sub_df = self.orig_df[self.orig_df[col_cond] == val]
                    spearancorr = sub_df[col_name_1].corr(sub_df[col_name_2], method='spearman')
                    if abs(spearancorr) < 0.995:
                        all_subsets_correlated = False
                        break

                    col_1_percentiles = sub_df[col_name_1].rank(pct=True)
                    col_2_percentiles = sub_df[col_name_2].rank(pct=True)

                    if spearancorr > 0:
                        subset_series = np.array([abs(x-y) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                    else:
                        subset_series = np.array([abs(x-(1.0 - y)) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])

                    for i_idx, i in enumerate(sub_df.index):
                        test_series[i] = subset_series[i_idx]

                if all_subsets_correlated:
                    self.__process_analysis_binary(
                        test_id,
                        self.get_col_set_name([col_name_1, col_name_2, col_cond]),
                        [col_name_1, col_name_2, col_cond],
                        test_series,
                        f'"{col_name_1}" is consistently to "{col_name_2}" when conditioning on "{col_cond}"')

    ####################################################################################
    # Data consistency checks for non-numeric columns in relation to all other columns
    ####################################################################################

    def __generate_dt_classifier(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self.__add_synthetic_column('dt cls. 1a', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('dt cls. 1b', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        arr = []
        for i in range(self.num_synth_rows):
            if self.synth_df['dt cls. 1a'][i] > 50:
                if self.synth_df['dt cls. 1b'][i] > 50:
                    arr.append('A')
                else:
                    arr.append('B')
            else:
                if self.synth_df['dt cls. 1b'][i] > 50:
                    arr.append('C')
                else:
                    arr.append('D')
        self.__add_synthetic_column('dt cls. 2', arr)
        self.__add_synthetic_column('dt cls. 3', self.synth_df['dt cls. 2'])
        if self.synth_df.loc[999, 'dt cls. 3'] == 'A':
            self.synth_df.at[999, 'dt cls. 3'] = 'B'
        else:
            self.synth_df.at[999, 'dt cls. 3'] = 'A'

    def __check_dt_classifier(self, test_id):
        """"
        Very similar to the test using a regression decision tree. The accuracy is measured in terms f1_score.
        """
        # todo: if there are many values, group the rare ones into "other"

        drop_features = []
        categorical_features = []
        for col_name in self.orig_df.columns:
            if not pandas_types.is_numeric_dtype(self.orig_df[col_name]):
                if self.orig_df[col_name].nunique() > 5:
                    drop_features.append(col_name)
                else:
                    categorical_features.append(col_name)

        for col_name in self.string_cols + self.binary_cols:
            if col_name in drop_features:
                continue
            clf = DecisionTreeClassifier(max_leaf_nodes=4)
            x_data = self.orig_df.drop(columns=set(drop_features + [col_name]))
            if len(x_data.columns) == 0:
                continue
            use_categorical_features = categorical_features.copy()
            if col_name in use_categorical_features:
                use_categorical_features.remove(col_name)
            x_data = pd.get_dummies(x_data, columns=use_categorical_features)
            for c in self.orig_df.columns:
                if c in x_data.columns:
                    x_data[c] = x_data[c].replace([np.inf, -np.inf, np.NaN], self.orig_df[c].median())

            y = self.orig_df[col_name]
            y = y.replace([np.inf, -np.inf, np.NaN], statistics.mode(y))
            y = y.astype(str)

            clf.fit(x_data, y)
            y_pred = clf.predict(x_data)
            f1_dt = metrics.f1_score(y, y_pred, average='macro')
            ft_naive = metrics.f1_score(y, [statistics.mode(y)] * len(y), average='macro')
            if (f1_dt > 0.8) and (f1_dt > (ft_naive * 1.5)):

                rules = tree.export_text(clf)
                cols = []
                for c_idx, c_name in enumerate(x_data.columns):
                    rule_col_name = f'feature_{c_idx}'
                    if rule_col_name in rules:
                        orig_col = c_name
                        for cat_col in categorical_features:
                            if c_name.startswith(cat_col):
                                orig_col = cat_col
                        cols.append(orig_col)
                        rules = rules.replace(rule_col_name, c_name)

                test_series = y == y_pred
                self.__process_analysis_binary(
                    test_id,
                    self.get_col_set_name(cols),
                    cols,
                    test_series,
                    f'All values in the column are consistently predictable based using a decision tree with the '
                    f'following rules: \n{rules}'
                )

    ####################################################################################
    # Data consistency checks for sets of four columns of any type
    ####################################################################################

    def __generate_two_pairs(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'two_pairs rand_a' and 'two_pairs rand_b' have equal values in the same rows as
            'two_pairs rand_c' and 'two_pairs rand_d', with 1 exception
        """
        self.__add_synthetic_column('two_pairs_rand_a', np.random.randint(0, 5, self.num_synth_rows))
        self.__add_synthetic_column('two_pairs_rand_b', np.random.randint(0, 5, self.num_synth_rows))
        self.__add_synthetic_column('two_pairs_rand_c', np.random.randint(0, 5, self.num_synth_rows))
        a_b_match_arr = [x == y for x, y, in zip(self.synth_df['two_pairs_rand_a'], self.synth_df['two_pairs_rand_b'])]
        self.__add_synthetic_column('two_pairs_rand_d',
                                    [x if y else x + 1 for x, y in zip(self.synth_df['two_pairs_rand_c'], a_b_match_arr)])
        self.synth_df.loc[999, 'two_pairs_rand_d'] = \
            self.synth_df.loc[999, 'two_pairs_rand_c'] + 1 if a_b_match_arr[999] \
                else self.synth_df.loc[999, 'two_pairs_rand_c']

    def __check_two_pairs(self, test_id):

        # Get the column type of each column
        column_types_dict = {}
        for col_name in self.binary_cols:
            column_types_dict[col_name] = 0  # Using integer codes for fast comparison
        for col_name in self.numeric_cols:
            column_types_dict[col_name] = 1
        for col_name in self.string_cols:
            column_types_dict[col_name] = 2
        for col_name in self.date_cols:
            column_types_dict[col_name] = 3

        is_na_dict = {}
        for col_name in self.orig_df.columns:
            is_na_dict[col_name] = self.orig_df[col_name].isna()

        sample_is_na_dict = {}
        for col_name in self.orig_df.columns:
            sample_is_na_dict[col_name] = self.sample_df[col_name].isna()

        match_okay_limit = self.num_rows / 10.0
        match_sample_okay_limit = len(self.sample_df) / 10.0
        for col_idx_1 in range(len(self.orig_df.columns)-3):
            col_name_1 = self.orig_df.columns[col_idx_1]
            if self.verbose >= 2:
                print(f"  Processing column {col_idx_1} of {len(self.orig_df.columns)} columns")
            for col_idx_2 in range(col_idx_1+1, len(self.orig_df.columns)):
                col_name_2 = self.orig_df.columns[col_idx_2]
                if column_types_dict[col_name_1] != column_types_dict[col_name_2]:
                    continue

                # Test on the sample data if col_name_1 and col_name_2 are often the same value
                match_1_2_sample_arr = [x == y for x, y, w, z in zip(self.sample_df[col_name_1], self.sample_df[col_name_2], sample_is_na_dict[col_name_1], sample_is_na_dict[col_name_2]) if (not w) and (not z)]
                if (match_1_2_sample_arr.count(True) < match_sample_okay_limit) or (match_1_2_sample_arr.count(False) < match_sample_okay_limit):
                    continue

                match_1_2_not_null_arr = [x == y for x, y, w, z in zip(self.orig_df[col_name_1], self.orig_df[col_name_2], is_na_dict[col_name_1], is_na_dict[col_name_1]) if (not w) and (not z)]
                if (match_1_2_not_null_arr.count(True) < match_okay_limit) or (match_1_2_not_null_arr.count(False) < match_okay_limit):
                    continue
                match_1_2_arr = [x == y for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])]
                for col_idx_3 in range(col_idx_1+1, len(self.orig_df.columns)-1):
                    if col_idx_3 == col_idx_2:
                        continue
                    col_name_3 = self.orig_df.columns[col_idx_3]
                    for col_idx_4 in range(col_idx_3+1, len(self.orig_df.columns)):
                        if col_idx_4 == col_idx_2:
                            continue
                        col_name_4 = self.orig_df.columns[col_idx_4]
                        if column_types_dict[col_name_4] != column_types_dict[col_name_3]:
                            continue

                        # Test on the sample data if col_name_3 and col_name_4 are often the same value
                        match_3_4_sample_arr = [x == y for x, y, w, z in zip(self.sample_df[col_name_3], self.sample_df[col_name_4], sample_is_na_dict[col_name_3], sample_is_na_dict[col_name_4]) if (not w) and (not z)]
                        if (match_3_4_sample_arr.count(True) < match_sample_okay_limit) or (match_3_4_sample_arr.count(False) < match_sample_okay_limit):
                            continue

                        match_3_4_not_null_arr = [x == y for x, y, w, z in zip(self.orig_df[col_name_3], self.orig_df[col_name_4], is_na_dict[col_name_3], is_na_dict[col_name_4]) if (not w) and (not z)]
                        if (match_3_4_not_null_arr.count(True) < match_okay_limit) or (match_3_4_not_null_arr.count(False) < match_okay_limit):
                            continue
                        match_3_4_arr = [x == y for x, y in zip(self.orig_df[col_name_3], self.orig_df[col_name_4])]
                        test_series = [x == y for x, y in zip(match_1_2_arr, match_3_4_arr)]
                        test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_3].isna() | self.orig_df[col_name_4].isna()
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name([col_name_1, col_name_2, col_name_3, col_name_4]),
                            [col_name_1, col_name_2, col_name_3, col_name_4],
                            test_series,
                            (f'Columns "{col_name_1}" and "{col_name_2}" have equal values in the same rows as '
                             f'"{col_name_3}" and "{col_name_4}"')
                        ) # todo: give stats about how often 1==2, and 3==4 and 1!=2.
                        # todo: when list exaples, give examples where match & where don't

    ####################################################################################
    # Data consistency checks for sets of columns of any type
    ####################################################################################
    def __generate_unique_sets_values(self):
        """
        Patterns without exceptions:
        Patterns with exception: 'unique_sets most_a' and 'unique_sets most_b' have 10 and 100 unique values
            respectively.
        """
        data_a = []
        for i in range(10):
            data_a.append([i]*100)
        data_a = np.array(data_a).reshape(1, -1)[0]

        data_b = []
        for _ in range(10):
            data_b.append(list(range(100)))
        data_b = np.array(data_b).reshape(1, -1)[0]

        self.__add_synthetic_column('unique_sets rand', data_a)
        self.__add_synthetic_column('unique_sets all_b', data_b)
        self.__add_synthetic_column('unique_sets most', data_b)
        self.synth_df.loc[999, 'unique_sets most'] = 98  # repeating a combination with 9 in column 'all_a'

    def __check_unique_sets_values(self, test_id):
        """
        """

        # Get the set of columns that have not too many unique values. This is not strictly necessary, but a set of
        # columns with a unique combination of values is less meaningful if one or more of the columns have many
        # unique values in themselves. Also get and cache the number of unique values per column.
        cols = []
        num_unique_vals_dict = {}
        for col_name in self.orig_df.columns:
            if self.orig_df[col_name].nunique() <= (self.num_rows / 2):  # todo: makes sense???
                cols.append(col_name)
                num_unique_vals_dict[col_name] = self.orig_df[col_name].nunique()

        found_any = False
        for subset_size in range(len(cols), 1, -1):
            if found_any:
                break

            calc_size = math.comb(len(cols), subset_size)
            skip_subsets = calc_size > self.max_combinations
            if skip_subsets:
                if self.verbose >= 2:
                    print((f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. max_combinations is "
                           f"currently set to {self.max_combinations:,}"))
                continue

            subsets = list(combinations(cols, subset_size))
            if self.verbose >= 2 and len(cols) > 15:
                print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset in subsets:
                max_combinations = 1
                for c in subset:
                    max_combinations *= num_unique_vals_dict[c]

                # If there are too few combinations to make unique combinations impossible (the number of combinations
                # is less than the number of rows), do not test. Also do not test if it will be unremarkable if there
                # are all unique combinations. The threshold for this is arbitrary, but set to a small multiple of the
                # number of rows.
                if self.num_rows <= max_combinations <= (self.num_rows * 2):
                    test_series = self.orig_df.duplicated(subset=subset)
                    num_dup = test_series.tolist().count(True)
                    if 0 < num_dup < self.contamination_level:
                        self.__process_analysis_binary(
                            test_id,
                            self.get_col_set_name(subset),
                            subset,
                            ~test_series,
                            f'The set {subset} consistently contain a unique combination of values',
                            ""
                        )
                        found_any = True
                        break

    ####################################################################################
    # Data consistency checks for complete rows of values
    ####################################################################################

    def __generate_missing_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 2 Null values (if only this test is tested),
            with the exception of 1 row
        """
        data_vals = [['a', 'b', None, None], ['a', None, None, 'd'], [None, 'b', None, 'd'], [None, None, 'c', 'd']]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('miss_per_row_0', data[0].values)
        self.__add_synthetic_column('miss_per_row_1', data[1].values)
        self.__add_synthetic_column('miss_per_row_2', data[2].values)
        self.__add_synthetic_column('miss_per_row_3', data[3].values)
        self.synth_df.loc[999, 'miss_per_row_0'] = None
        self.synth_df.loc[999, 'miss_per_row_1'] = None
        self.synth_df.loc[999, 'miss_per_row_2'] = None
        self.synth_df.loc[999, 'miss_per_row_3'] = None

    def __check_missing_values_per_row(self, test_id):
        test_series = self.orig_df.isna().sum(axis=1)
        self.__process_analysis_counts(
            test_id,
            self.get_col_set_name(list(self.orig_df.columns)),
            list(self.orig_df.columns),
            test_series,
            "The values in the dataset consistently have",
            " Null values per row"
        )

    def __generate_zero_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 2 zero values (if only this test is tested),
            with the exception of 1 row
        """
        data_vals = [[1, 2, 0, 0], [1, 0, 0, 4], [0, 2, 0, 4], [0, 0, 3, 4]]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('zero_per_row_0', data[0].values)
        self.__add_synthetic_column('zero_per_row_1', data[1].values)
        self.__add_synthetic_column('zero_per_row_2', data[2].values)
        self.__add_synthetic_column('zero_per_row_3', data[3].values)
        self.synth_df.loc[999, 'zero_per_row_0'] = 0
        self.synth_df.loc[999, 'zero_per_row_1'] = 0
        self.synth_df.loc[999, 'zero_per_row_2'] = 0
        self.synth_df.loc[999, 'zero_per_row_3'] = 0

    def __check_zero_values_per_row(self, test_id):
        test_series = self.orig_df.applymap(lambda x: (x is None) or (x == 0)).sum(axis=1)
        self.__process_analysis_counts(
            test_id,
            self.get_col_set_name(list(self.orig_df.columns)),
            list(self.orig_df.columns),
            test_series,
            "The values in the dataset consistently have",
            " elements with value 0 per row"
        )

    def __generate_unique_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 3 unique values (if only this test is
            tested), with the exception of 1 row
        """
        data_vals = [[1, 1, 3, 4], [1, 6, 6, 4], [0, 1, 0, 5], [4, 0, 3, 4]]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('unique_per_row_0', data[0].values)
        self.__add_synthetic_column('unique_per_row_1', data[1].values)
        self.__add_synthetic_column('unique_per_row_2', data[2].values)
        self.__add_synthetic_column('unique_per_row_3', data[3].values)
        self.synth_df.loc[999, 'unique_per_row_0'] = 1
        self.synth_df.loc[999, 'unique_per_row_1'] = 2
        self.synth_df.loc[999, 'unique_per_row_2'] = 3
        self.synth_df.loc[999, 'unique_per_row_3'] = 4

    def __check_unique_values_per_row(self, test_id):
        test_series = self.orig_df.apply(lambda x: len(set(x)), axis=1)
        self.__process_analysis_counts(
            test_id,
            self.get_col_set_name(list(self.orig_df.columns)),
            list(self.orig_df.columns),
            test_series,
            "The values in the dataset consistently have",
            " unique values per row"
        )

    def __generate_negative_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 2 negative values (if only this test is
            tested), with the exception of 1 row
        """
        data_vals = [[-1, -2, 0, 0], [-1, 0, 0, -4], [1, -2, 0, -4], [0, 2, -3, -4]]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self.__add_synthetic_column('neg_per_row_0', data[0].values)
        self.__add_synthetic_column('neg_per_row_1', data[1].values)
        self.__add_synthetic_column('neg_per_row_2', data[2].values)
        self.__add_synthetic_column('neg_per_row_3', data[3].values)
        self.synth_df.loc[999, 'neg_per_row_0'] = 0
        self.synth_df.loc[999, 'neg_per_row_1'] = 0
        self.synth_df.loc[999, 'neg_per_row_2'] = 0
        self.synth_df.loc[999, 'neg_per_row_3'] = 0

    def __check_negative_values_per_row(self, test_id):
        test_series = self.orig_df.applymap(lambda x: isinstance(x, numbers.Number) and x < 0).sum(axis=1)
        self.__process_analysis_counts(
            test_id,
            self.get_col_set_name(list(self.orig_df.columns)),
            list(self.orig_df.columns),
            test_series,
            "The values in the dataset consistently have",
            " negative values per row"
        )

    def __generate_small_avg_rank_per_row(self):
        """
        Patterns without exceptions: None. This test does not flag patterns.
        Patterns with exception: Row 999, over all numeric columns, has values with, on average, low percentiles.
        """
        for i in range(10):
            self.__add_synthetic_column(f'small_avg_rank_rand_{i}', [random.random() for _ in range(self.num_synth_rows -1)] + [0.000001])

    def __check_small_avg_rank_per_row(self, test_id):
        rand_df = pd.DataFrame()
        for col_name in self.numeric_cols:
            rand_df[col_name] = self.orig_df[col_name].rank(pct=True)
        rand_df['Avg Percentile'] = rand_df.mean(axis=1)

        d1 = rand_df['Avg Percentile'].quantile(0.1)
        d9 = rand_df['Avg Percentile'].quantile(0.9)
        idr = abs(d9 - d1)
        lower_limit = d1 - (self.idr_limit * idr)
        test_series = rand_df['Avg Percentile'] >= lower_limit
        flagged_vals = [x for x in rand_df['Avg Percentile'] if x < lower_limit]
        self.__process_analysis_binary(
            test_id,
            self.get_col_set_name(self.numeric_cols),
            self.numeric_cols,
            test_series,
            f"The row contains values that are consistently in the lower percentiles for their columns",
            display_info={"percentiles": rand_df['Avg Percentile'].sample(n=min(len(rand_df), 1000)),
                          "flagged_vals": flagged_vals}
        )

    def __generate_large_avg_rank_per_row(self):
        """
        Patterns without exceptions: None. This test does not flag patterns.
        Patterns with exception: Row 999, over all numeric columns, has values with, on average, low percentiles.
        """
        for i in range(10):
            self.__add_synthetic_column(f'large_avg_rank_rand_{i}', [random.random() for _ in range(self.num_synth_rows -1)] + [2.0])

    def __check_large_avg_rank_per_row(self, test_id):
        rand_df = pd.DataFrame()
        for col_name in self.numeric_cols:
            rand_df[col_name] = self.orig_df[col_name].rank(pct=True)
        rand_df['Avg Percentile'] = rand_df.mean(axis=1)

        q1 = rand_df['Avg Percentile'].quantile(0.25)
        q3 = rand_df['Avg Percentile'].quantile(0.75)
        iqr = abs(q3 - q1)
        # As the value is limited to 1.0, we do not use self.iqr_limit here, but instead the standard coefficient of
        # 2.2 for testing for outliers.
        upper_limit = q3 + (2.2 * iqr)
        test_series = rand_df['Avg Percentile'] <= upper_limit
        flagged_vals = [x for x in rand_df['Avg Percentile'] if x > upper_limit]
        self.__process_analysis_binary(
            test_id,
            self.get_col_set_name(self.numeric_cols),
            self.numeric_cols,
            test_series,
            f"The row contains values that are consistently in the upper percentiles for their columns",
            display_info={"percentiles": rand_df['Avg Percentile'].sample(n=min(len(rand_df), 1000)),
                          "flagged_vals": flagged_vals}
        )

def get_num_digits(num):
    num_str = str(np.format_float_positional(num))
    if num_str.count('.') == 0:
        return 0
    digits_str = num_str.split('.')[1]
    len_digits_str = len(digits_str)
    if len_digits_str == 0:
        return 0
    digits_val = int(digits_str)
    if digits_val == 0:
        return 0
    return len_digits_str


def styling_orig_row(x, row_idx, flagged_arr):
    df_styler = pd.DataFrame('', index=x.index, columns=x.columns)
    for c_idx, c_flagged in enumerate(flagged_arr):
        if c_flagged:
            df_styler.iloc[row_idx, c_idx+1] = 'background-color: #cfc64a; color: black'
        else:
            df_styler.iloc[row_idx, c_idx+1] = 'background-color: #e5f8fa; color: black'
    return df_styler


def styling_flagged_rows(x, flagged_cells, flagged_arr):
    print("flagged_arr: ", flagged_arr)
    print("flagged_cells: ", flagged_cells)
    print("flagged_cells: ", flagged_cells[0])

    df_styler = pd.DataFrame('background-color: #e5f8fa; color: black', index=x.index, columns=x.columns)
    for row_idx in x.index:
        for col_idx, col_name in enumerate(x.columns[:-1]):  # Do not check the 'FINAL SCORE' column
            if flagged_cells[row_idx, col_idx] > 0:
                df_styler.loc[row_idx][x.columns[col_idx]] = 'background-color: #cfc64a; color: black'
    return df_styler


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def is_missing(x):
    # todo: check NaD (not a date) too. and NaT (not a time)
    if x is None:
        return True
    if isinstance(x, numbers.Number):
        return math.isnan(x)
    if x != x:
        return True
    if type(x) == str:
        return x == ""
    if type(x) == list:
        return len(x) == 0
    return False


def get_subsets(arr, min_size=1):
    subsets = []
    [subsets.extend(list(combinations(arr, n))) for n in range(len(arr)+1, min_size-1, -1)]
    return subsets


def array_to_str(arr):
    arr = sorted(arr)
    arr_str = ""
    for v in arr:
        arr_str += str(v) + ", "
    arr_str = arr_str[:-2]
    return arr_str


def replace_special_with_space(x):
    if x is None:
        return ""
    if x in [np.inf, -np.inf, np.NaN]:
        return ""
    return ''.join([c if ((c in string.ascii_letters) or (c in string.digits)) else " " for c in x])


def call_test(dc, test_id):
    print(test_id)