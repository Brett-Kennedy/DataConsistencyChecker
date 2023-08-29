# DataConsistencyChecker
A python tool to examine datasets for consistency. It performs approximately 150 tests, identifying patterns in the data and any exceptions to these. The tool provides useful analysis, which may be used for any EDA (Exploratory Data Analysis) work, and may be useful for interpretable outlier detection. The tool may be run on any tabular dataset.

## Background

The central idea of the tool is to automate the tests that would be done by a person examining a dataset and trying to determine the patterns within the columns, between the columns, and between the rows (in cases where there is some meaning to the order of the rows). Intuitively, the tool essentially does what a person, studying a new dataset, would do, but automatically. This does not remove the need to do manual data exploration, to run other EDA tools, and to, where the goal is identifying outliers, run other forms of outlier detection. But, the tool does cover much of the work identifying the patterns in the data and the exceptions to these, with the idea that these are two of the principle tasks to understand a dataset. The tool allows users to perform this work quicker, more exhaustively and consistently, and covering more tests than would normally be done. 

The tool executes a large set of tests over a dataset. Each of the tests is run over either single columns (for example, checking for rare, unsually small or large values, etc), over pairs of columns (for example checking if one column is consistently larger than the other, contains similar characters (in the case of code or ID string values), etc.), or over larger sets of columns (for example, checking if one column tends to be the sum, or mean, of another set of columns). As an example, one test examines the number of decimal digits typically found in each numeric column. If a column consistently contains, say, four decimal digits, with no exceptions, this will be reported as a strong pattern without exceptions; if the data in this column nearly always has four decimal digits, with a small number of expections (for example having eight decimal digits), this will be reported as a strong pattern with exceptions. In this example, this suggests the identified rows may have been collected or processed in a different manner than the other rows. And, while this may not be interesting in itself, where rows are flagged multiple times for this or other issues, users may become progressively more confident that the repeatedly-flagged rows are, in fact, in some sense different. 

While all of the individual tests are straight-forward (and therefore interpretable) there are real advantages to running them within a single package, notably the ability to identify rows that are significantly different from the majority, even where this is only evident from multipe subtle deviations. Running a large set of tests fascilitates this, as it tends to pick up well where some instances are atypical, even where this is in a sense not normally tested. Further, running many tests allows for an inuitive scoring method, as each row is scored simply based on the number of times it has been flagged. Any rows with high scores were, then, flagged many times for different exceptions and are therefore quite reasonably likely to be true outliers. 

Another signicant advantage of running multiple tests in a single package is the ability to ammortize processing work over many tests. Much of the computation necessary for the tests is shared among two or more tests and consequently running many tests on the same dataset can result in increased performance, in terms of time per test. 

### EDA

DataConsistencyChecker may be used for exploratory data analsys simply by running the tool and examining the patterns and exceptions to the patterns that are found. For EDA purposes, the patterns found may be equally of interest, regardless of whether exceptions to the patterns were found or not, but often the fact that exceptions exist or do not is itself relevant to understanding the patterns in the data, and therefore the data itself. 

Exploratory data analysis may be run for a variety of purposes, but generally most relate to one of two high-level purposes: ensuring the data is of sufficient quality to build a model (or can be made to be of sufficient quality after removing any necessary rows or columns, or replacing any necessary individual values), and gaining insights into the data in order to help build appropriate models. This tool may be of assistence for both these purposes.  

The tool performs many tests not typical of most EDA tools, which tend to look at data distributions, correlations, missing values, and some other concerns.  While many tests in this package have some overlap with other EDA tools, the majority do not, and DataConsistencyChecker can compliment outher EDA tools well.  

### Outlier Detection

The majority of outlier detectors work either on strictly numeric data, or on strictly categorical data, with numeric outlier detectors seeking to identify unusual rows as points in high-dimensional space far from the majority of other rows, and with categorical outlier detectors seeking to identify unusual combinations of values. Outlier detectors that assume numeric data require encoding categorical columns, and detectors that assume categorical data require binning the numeric columns. These can work well, but some signal is lost in both cases. DataConsistencyChecker handles categorical, numeric, and date/time data equally, without requiring encoding or binning data, specifiying distance metrics, or other pre-processing required by many detectors. DataConsistencyChecker's tests each run on the original, unprocessed data, taking advantage of this to find often more subtle anomalies in the data. 

A major drawback, however, of most outlier detectors is they can be very much blackboxes. The detectors are very useful, but can be uninterpretable, especially with high-dimensional datasets: it can be difficult to confirm the rows are more unusual than the majority of rows or to even determine why they were flagged as outliers. DataConsistencyChecker's tests are each interpretable, and the final scoring system is completely transparent. 

Running the tool, it's common for rows that are unusual to be flagged multiple times. This allows users to evaluate the outlierness of any given row based on the number of times it was flagged for issues, relative to the other rows in the dataset. For example, if a row contains some values that are unusually large, as well as some strings with unusual characters, strings of unusual lengths, time values at unusual times of day, and rare values in categorical columns, the row will then be flagged multiple times, giving it a relatively high outlier score. 

As this tool is limited to interpretable outlier detection methods, it does not test for the multi-dimensional outliers detected by other algorithms such as Isolation Forest, Local Outlier Factor, Angle-based Outlier Detection, Cluster-based outlier detection, etc. These outlier detection algorithms should generally also be executed to have a full understanding of the outliers present in the data. 

The unusual data found may be due to data collection errors, mixing different types of data together, or other issues that may be considered errors, or that may be informative. Some may point to forms of feature engineering, which may be useful for downstream tasks. 


## Intallation
The code consists of a single [python file](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/check_data_consistency.py) which may be downloaded and included in any project. It does not rely on any other packages than numpy, pandas, scipy, matplotlib, seaborn, and other standard libraries. Once downloaded, it may be included as:

```python
from check_data_consistency import DataConsistencyChecker
```

## Getting Started
```python
import pandas as pd
import sklearn.datasets as datasets
from check_data_consistency import DataConsistencyChecker

data = datasets.fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

dc = DataConsistencyChecker()
dc.init_data(df)
dc.check_data_quality()
dc.display_detailed_results()
```

It is necessary to first instantiate a DataConsistencyChecker object, call init_data() with the data to be cheked as a pandas dataframe, and call check_data_quality(). After this is complete, there are a number of additional APIs available to examine the results and assess the findings, including APIs to summarize the patterns and exceptions by row, column, and test, to describe the most-flagged rows, and get the total score per row. In this example, display_detailed_results() is called, which displays each pattern and each exception found in detail, including examples of values for the relevant columns flagged and not flagged, and plots where possible. 

## Example Notebooks

**APIs Demo**

The [APIs Demo](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_APIs.ipynb) notebook provides examples of many of the APIs provided with the tool, though many of more common APIs are covered by the California Housing and Breast Cancer demo notebooks, and are not covered here. This notebook goes through an example with the Boston Housing dataset. Note, in some cases the background coloring in the displayed notebooks will not render in github.

Each of the notebooks provides examples of some of the plots available. 

Example display of a row identified as an outler:

![example](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/images/img1.jpg)

<br>

**Hypothyroid Demo**

The [Hypothyroid Demo](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_Multiple_Executions.ipynb) notebook is a simple example, examining a dataset, getting a list of the patterns and exceptions found, and getting more detail on a subset of these that appear most interesting.

<br>

**California Housing Demo**

The [California Housing Demo](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_California_Housing.ipynb) notebook goes through a more typical example examining a dataset. This focusses on the quick_report() API, which is a convenience method wrapping several other APIs, to give an overview of the findings. After this, display_detailed_results() is called to provide more information on specific issues flagged.  

<br>

**Breast Cancer Demo**

The [Breast Cancer Demo](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_Real_Breast_Cancer.ipynb) notebook goes though another typical example of examinging a dataset, calling somewhat different APIs than the [California Housing Demo](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_California_Housing.ipynb) example. 

<br>

**Multiple Exectutions Demo**

The [Multiple Exectutions Demo](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_Multiple_Executions.ipynb) notebook demonstrates examples of working with the tool in a couple different ways: 1) where a set of tests are run, then another set of tests are run, replacing the first results, and another case where the additional tests append to the set of results found, gradually buiding up a complete set, potentially for further anaysis, or for a final report. 

<br>

**OpenML Demo**

The [OpenML](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_OpenML_Datasets.ipynb) notebook runs DataConsistencyChecker on several datasets from OpenML, and displays a small number of findings for each. In most cases, other patterns were also identified, which may or may not be interesting. In general, when using DataConsistencyChecker, there may be a step involved with examining the patterns discovered to identify the relevant ones, but this is typically quite quick and worthwhile to identify the interesting patterns. APIs are provided to assist with processessing patterns where many are discovered, with examples in the other notebooks.

This notebook, for each dataset, runs the checker for a small number of tests, then displays some subset of the results found, often filtering the results to show only a single issue, or the issues related to a single feature. Many more patterns are found in each of these, but the purpose of the notebook is to provide examples of some of the patterns that can be found relatively often. 

The tests are able to find instances where columns are correlated, where columns have identical values, where one column is equal, or approximately equal to the sum, product, ratio, or difference in two other coumns, where one column may be predicted from other columns using a linear regression or small decision tree, where columns are ordered with monotonically increasing values, or with cyclical patterns of values. 

An example where two columns were found to be correlated with an exception:


![example](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/images/img2.jpg)

An example where one column was found to be the sum of two other columns with two exceptions:


![example](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/images/img3.jpg)


## Performance
The tool typically runs in under a minute for small files and under 30 minutes even for very large files, with hundreds of thousands of rows and hundreds of columns. However, it is often useful to specify to execute the tool quicker, especially if calling it frequently, or with many datasets. Several techniques to speed the execution are listed here:

- Exclude some tests. Some may be of less interest to your project or may be slower to execute.  
- Set the max_combinations parameter to a lower value, which will skip a significant amount of processing in some cases. The default is 100,000. The value relates to the number of sets of columns examined. For example, if the test checks each triple of columns, such as with SIMILAR_TO_RATIO, then the number of sets of three numeric columns is the number of combinations. This is equal to (N choose 3), where N is the number of numeric columns. With verbose set to 2 or higher, it's possible to see the number of combinations required and the current setting of max_combinations. 
- Run on a sample of the rows. Doing this, you will not be able to properly identify exceptions to the patterns, but will be able to identify the patterns themselves. 
- Columns may be removed from the dataframe before running the tool to remove checks on these columns. 
- The fast_only parameters in check_data_quality() may be set to True. Setting this, only tests that are consistently fast will be executed.
- Set the verbose level to 2 or 3. This allows users to monitor the progress better and determine if it is acceptable to wait, or if any of the other steps listed here should be tried.

## Date Columns

It is recommended that the pandas dataframe passed has any date columns set such that the dtype is datetime64 or timestamp. However, the tool will attempt to convert any columns with dtype as object to datetime, and will treat any that can be successfully converted as datetime columns, which means all tests related to date or time values will be executed on these columns. If verbose is set to 2 or higher, the column data types used will be displayed at the top ot the output running check_data_quality(). It is also possible to call display_columns_types_list() or display_columns_types_table() to check the data types that were inferred for the passed dataframe. If desired, you may pass a list of date columns to init_data() and the tool will attempt to treat all of these, and only these, columns as datetime columns. 

## Data Dredging and False Postives

This tool runs a large number of tests, some of which check many sets of columns. For example, the test to check if one column tends to be the sum of another set of columns, can check a very large number of subsets of the full set of numeric columns. This can lead to false positives, where patterns are reported that are random, or not meaningful. Ideally the output is clear enough that any patterns found may be confirmed, and checked to determine how relevant they are. A key property of the tool, though, is that each row is tested many times, and so while there may be some false positives reported, rows flagged many times, particularly those reported many times for different tests on different columns, are certainly unusual. Note though, reliably identifying rows as unusual does not suggest anything regarding whether the rows are correct or if they are useful, only that they are different from the other rows in the same dataset. 

## Internal Parameters

As with all outlier detectors, there are some thresholds and other decisions necessary to define what constitutes a pattern and what constitutes an exception to the pattern. There are options to configure the tests, such as the coefficients used for interquartile range and interdecile range tests. For the most part, though, we have taken the approach of trying to limit options, and to provide clear, sensible tests, without the need to experiment or run the tool multiple times. However, almost all tests could, at least potentially, be tweaked to run with different thresholds or with different options, which would produce different output. For example, one test exists to check if, for sets of numeric columns, if the values in these columns, row for row, tend to all be positive or all negative together. In this case, there is a decision how to handle zero values. While such decisions are made based on testing on a very large number of datasets, other decisions could be made. We have attempted in all cases to make the most reasonable decisions and, though alternative, similar tests may also be useful, the output here has been confirmed to be accurate and useful.

As another example, there are tests to determine if the values in one column can be predicted from the values in the other columns using a small decision tree or linear regression. In these cases, there are decisions related, for example, to the minimum accuracies of the models. As such, these tests should be taken simply as presenting the patterns it does find, without any statement about other patterns that could be found were different options used. Configuring these options does invite running through many combinations of options, increases data dredging and producing less combrehensible output. 

## Clearing Issues

All outlier detectors, including DataConsistencyChecker, allow some tuning. Detectors that return binary labels further require some thresholds, but all provide some options. For example, kth Nearest Neighbors requires defining k, a distance metric, a means of scaling each numeric column, a method of encoding categorical features, and so on. This is true of DataQualityChecker as well, though it has the advantage of executing many tests, such that any decisions made on individual tests will tend to average out over many tests, therefore requiring less tuning, though some tuning is possible if desired. Primarily though, we recommend tuning in the form of clearing issues, such that analysis begins with the full set of issues identified, followed by some manual pruning of any issues considered inconsequental, leaving the remaining issues, and a final outlier score per row. 

A set of APIs, clear_results(), and restore_results() are provided to allow users to examine the results, and incrementally clear any issues found which may be considered non-issues, leaving a final set of patterns and exceptions that are consisidered noteworthy. This can help reduce some noise when a non-trivial number of issues are found, or when reporting issues to others. 

An example is provided in [clearing issues demo notebook](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_Clear_Issues.ipynb)

## Contamination Level

While exceptions to patterns necessarily require decisions related to thresholds, patterns without exceptions are more straight-forward. In most cases, the pattern clearly exists or does not. For example a numeric column is monotonically increasing or it is not; a string column consistently contains 4 uppercase characters or it does not. These are typically clear, and can be quite straightforward to assess. With exceptions, however, some decisions must be made about the nature of the exceptions, but in most cases, this is simply a case of specifiying the number of exceptions permitted such that we recognize a pattern as still existing, albeit with exceptions. 

In general, much of tuning outlier detectors is in the form of determining how many outliers should be flagged. With DataConsistencyChecker, this is done most simply with the contamination_level parameter, which defines the maximum number or exceptions to allow to a pattern for it to be considered a pattern with exceptions, as opposed to a non-pattern. This may be specified as an absolute count or as a fraction of the total number of rows. The default is set to 0.5%, which we have found quite sensible, but tuning this may be quite reasonable as well. 

If there are, say, 10,000 rows in a dataset, then the default value of 0.5% exceptions to any pattern allows here 50 rows. For example, one test checks if two columns have values such that one is the result of rounding the other. In this example, if this is true in all 10,000 rows, this will be identified as a pattern with no exceptions; if this is true in all but 1 to 50 rows, this will be considered a pattern with exceptions, and if false in 51 or more rows, it will not be considered a pattern. Adjusting the contamination_level up or down can provide a more suitable number of exceptions for some cases. 

In most cases, where tuning is necessary, tuning this single parameter will be sufficient. However, it you wish to identify more outliers than have been flagged, it is also possible to increase max_combinations, which will allow examining of more combinations of features for some tests. 

## Noting the Patterns Not Found

DataConsistencyChecker performs sufficient tests that it's reasonably likely to find something of interest in any non-trivial dataset and, where it does not, there is real meaning in the fact that it does not -- this suggests a very consistent dataset. 

In general, what it does not flag can be of interest as well as what it does flag. For example, any rows with scores of zero may be reasonably considered to be very normal with respect to the dataset. As well, the tool provides interfaces to identify where patterns were not present. For example, if it may be expected that a column contains entirely positive values, then it is possible to see that it is not the case that this column contains entirely positive values (the POSITIVE test did not identify a pattern for this column), or that it contains almost entirely positive values, with some exceptions.  

## Types of Patterns Presented
We may look at each test as covering one of three cases related to patterns: 1) The test may identify patterns that do not occur in most datasets and are interesting in themselves, even where no exceptions are found, for example where one column is found to be the sum of two or more other columns, or two columns have identical values. 2) The test may identify patterns that are common in datasets, and may only be interesting if there are exceptions, for example finding that all numeric values are positive, or that all string values are strictly in lowercase. These, by default, are not returned in most API calls, though setting show_short_list_only to False for the relevant APIs will present these patterns as well. 3) Some tests have no concept of patterns, for example, tests to find very small or very large values; these tests return, if there are any results, only exceptions. All tests may return exceptions. 

## Synthetic Data
The tool provides an API to generate synthetic data, which has been designed to provide columns that will be flagged by the tests, such that each test flags at least one column or set of columns. This may be useful for new users to help understand the tests, as they provide useful examples, and may be altered to determine when specifically the tests flag or do not flag patterns. The synthetic data is also used by the unit tests to help ensure consistent behaviour as the tests are expanded and improved. 

## Unit Tests
The unit tests are organized such that each test .py file tests one DataQualityChecker test, both on real and synthetic data. 

### Unit Tests on Synthetic data with null values
As null values are common in real-world data, each test must be able to handle null values in a defined manner. For most tests, the null values do not support or violate the patterns discovered. 

### Synthetic data on all columns vs those specific to the tests
When generating synthetic data, a set of columns, typically two to five columns, will be created in order to exercise the related test. Each set of columns is relevant to only one test, though, it created, will increase the test coverage of the other tests. For example, to test FEW_WITHIN_RANGE, the synthetic data creates three columns: 'few in range rand', 'few in range all', and ''few in range most'. 

It is possible to call check_data_quality() on either a specific set of tests, or without specifying a set of tests, which will result in all tests being executed. Where only a subset of the available tests are executed, it may be desired to generate only the columns relevant to these tests. This will result in faster execution times and more predictable results. It is also possilble to specify to create all synthetic columns, including those designed for other tests. These columns will be covered as well by any tests executed and will provide more testing. Typically any columns flagged by a test will be flagged regardless if other synthetic columns are generated or not. 

## Tests
```
MISSING_VALUES:                Check if all values in a column are consistently present / consistently missing.
RARE_VALUES:                   Check if there are any rare values in a column.
UNIQUE_VALUES:                 Check if there are consistently unique values with a column.
PREV_VALUES_DT:                Check if the values in a column can be predicted from previous values in that column using
                                 a simple decision tree.
MATCHED_MISSING:               Check if two columns have missing values consistently in the same rows.
UNMATCHED_MISSING:             Check if two columns both frequently have null values, but consistently not in the same rows.
SAME_VALUES:                   Check if two columns consistently have the same values.
SAME_OR_CONSTANT:              Check one column consistently has either the same value as another column, or a small
                                 number of other values.
POSITIVE:                      Check if all numbers in a column are positive.
NEGATIVE:                      Check if all numbers in a column are negative.
NUMBER_DECIMALS:               Check if there is a consistent number of decimal digits in each value in a column.
RARE_DECIMALS:                 Check if there are any uncommon sets of digits after the decimal point.
COLUMN_ORDERED_ASC:            Check if a column is monotonically increasing.
COLUMN_ORDERED_DESC:           Check if a column is monotonically decreasing.
COLUMN_TENDS_ASC:              Check if a column is generally increasing.
COLUMN_TENDS_DESC:             Check if a column is generally decreasing.
SIMILAR_PREVIOUS:              Check if all values are similar to the previous value in the column, relative to the range
                                 of values in the column.
UNUSUAL_ORDER_MAGNITUDE:       Check if there are any unusual numeric values, in the sense of having an unusual order of
                                 magnitude for the column.
FEW_NEIGHBORS:                 Check if there are any unusual numeric values, in the sense of being distant from both the
                                 next smallest and next largest values within the column.
FEW_WITHIN_RANGE:              Check if there are any unusual numeric values, in the sense of having few other values in
                                 the column within a small range.
VERY_SMALL:                    Check if there are any very small values relative to their column.
VERY_LARGE:                    Check if there are any very large values relative to their column.
VERY_SMALL_ABS:                Check if there are any very small absolute values relative to its column.
MULTIPLE_OF_CONSTANT:          Check if all values in a column are multiples of some constant.
ROUNDING:                      Check if all values in a column are rounded to the same degree.
NON_ZERO:                      Check if all values in a column are non-zero.
LESS_THAN_ONE:                 Check if all values in a column are between -1.0 and 1.0, inclusive.
GREATER_THAN_ONE:              Check if all values in a column are less than -1.0 or greater than 1.0, inclusive.
INVALID_NUMBERS:               Check for values in numeric columns that are not valid numbers, including values that
                                 include parenthesis, brackets, percent signs and other values.
LARGER:                        Check if one column is consistently larger than another, but within one order of
                                 magnitude.
MUCH_LARGER:                   Check if one column is consistently at least one order of magnitude larger than another.
SIMILAR_WRT_RATIO:             Check if two columns are consistently similar, with respect to their ratio, to each other.
SIMILAR_WRT_DIFF:              Check if two columns are consistently similar, with respect to absolute difference, to
                                 each other.
SIMILAR_TO_INVERSE:            Check if one column is consistently similar to the inverse of another column.
SIMILAR_TO_NEGATIVE:           Check if one column is consistently similar to the negative of another column.
CONSTANT_SUM:                  Check if the sum of two columns is consistently similar to a constant value.
CONSTANT_DIFF:                 Check if the difference between two columns is consistently similar to a constant value.
CONSTANT_PRODUCT:              Check if the product of two columns is consistently similar to a constant value.
CONSTANT_RATIO:                Check if the ratio of two columns is consistently similar to a constant value.
EVEN_MULTIPLE:                 Check if one column is consistently an even integer multiple of the other.
RARE_COMBINATION:              Check if two columns have any unusual pairs of values.
CORRELATED_FEATURES:           Check if two columns are consistently correlated.
MATCHED_ZERO:                  Check if two columns have a value of zero consistently in the same rows.
OPPOSITE_ZERO:                 Check if two columns are consistently such that one column contains a zero and the other
                                 contains a non-zero value.
RUNNING_SUM:                   Check if one column is consistently the sum of its own value from the previous row and
                                 another column in the current row.
A_ROUNDED_B:                   Check if one column is consistently the result of rounding another column.
MATCHED_ZERO_MISSING:          Check if two columns consistently have a zero in one column and a missing value in the
                                 other.
SIMILAR_TO_DIFF:               Check if one column is consistently similar to the difference of two other columns.
SIMILAR_TO_PRODUCT:            Check if one column is consistently similar to the product of two other columns.
SIMILAR_TO_RATIO:              Check if one column is consistently similar to the ratio of two other columns.
LARGER_THAN_SUM:               Check if one column is consistently larger than the sum of two other columns.
SUM_OF_COLUMNS:                Check if one column is consistently similar to the sum of two or more other columns.
MEAN_OF_COLUMNS:               Check if one column is consistently similar to the mean of two or more other columns.
MIN_OF_COLUMNS:                Check if one column is consistently similar to the minimum of two or more other columns.
MAX_OF_COLUMNS:                Check if one column is consistently similar to the maximumn of two or more other columns.
ALL_POS_OR_ALL_NEG:            Identify sets of columns where the values are consistently either all positive, or all
                                 negative.
ALL_ZERO_OR_ALL_NON_ZERO:      Identify sets of columns where the values are consistently either all zero, or non-zero.
DECISION_TREE_REGRESSOR:       Check if a numeric column can be derived from the other columns using a small decision
                                 tree.
LINEAR_REGRESSION:             Check if a numeric column can be derived from the other numeric columns using linear
                                 regression.
EARLY_DATES:                   Check for dates significantly earlier than the other dates in the column.
LATE_DATES:                    Check for dates significantly later than the other dates in the column.
UNUSUAL_DAY_OF_WEEK:           Check if a date column contains any unusual days of the week.
UNUSUAL_DAY_OF_MONTH:          Check if a date column contains any unusual days of the month.
UNUSUAL_MONTH:                 Check if a date column contains any unusual months of the year.
UNUSUAL_HOUR:                  Check if a datetime / time column contains any unusual hours of the day. This and
                                 UNUSUAL_MINUTES also identify where it is inconsistent if the time is included in the
                                 column.
UNUSUAL_MINUTES:               Check if a datetime / time column contains any unusual minutes of the hour. This and
                                 UNUSUAL_MINUTES also identify where it is inconsistent if the time is included in the
                                 column.
CONSTANT_DOM:                  Check if a date column spans multiple months and the values are consistently the same day
                                 of the month
CONSTANT_LAST_DOM:             Check if a date column spans multiple months and the values are consistently the last day
                                 of the month
CONSTANT_GAP:                  Check if there is consistently a specific gap in time between two date columns.
LARGE_GAP:                     Check if there is an unusually large gap in time between dates in two date columns.
SMALL_GAP:                     Check if there is an unusually small gap in time between dates in two date columns.
LATER:                         Check if one date column is consistently later than another date column.
SAME_DATE:                     Check if two date columns consistently contain the same date, but may have different
                                 times.
SAME_MONTH:                    Check if two date columns consistently contain the same month, but may have different days
                                 or times.
LARGE_GIVEN_DATE:              Check if a numeric value is very large given the value in a date column.
SMALL_GIVEN_DATE:              Check if a numeric value is very small given the value in a date column.
BINARY_SAME:                   For each pair of binary columns with the same set of two values, check if they
                                 consistently have the same value.
BINARY_OPPOSITE:               For each pair of binary columns with the same set of two values, check if they
                                 consistently have the opposite value.
BINARY_IMPLIES:                For each pair of binary columns with the same set of two values, check if when one has a
                                 given value, the other consistently does as well, though the other direction may not be
                                 true.
BINARY_AND:                    For sets of binary columns with the same set of two values, check if one column is
                                 consistently the result of ANDing the other columns.
BINARY_OR:                     For sets of binary columns with the same set of two values, check if one column is
                                 consistently the result of ORing the other columns.
BINARY_XOR:                    For sets of binary columns with the same set of two values, check if one column is
                                 consistently the result of XORing the other columns.
BINARY_NUM_SAME:               For sets of binary columns with the same set of two values, check if there is a consistent
                                 number of these columns with the same value.
BINARY_RARE_COMBINATION:       Check for rare sets of values in sets of three or more binary columns.
BINARY_MATCHES_VALUES:         Check if the binary column is consistently one value when the values in a numeric column
                                 have low values, or when they have high values.
BINARY_TWO_OTHERS_MATCH:       Check if a binary column is consistently one value when two other columns have the same
                                 value as each other.
BINARY_MATCHES_SUM:            Check if the binary column is consistently true when the sum of a set of  numeric columns
                                 is over some threshold.
BLANK_VALUES:                  Check for blank strings and values that are entirely whitespace.
LEADING_WHITESPACE:            Check for strings with unusual leading whitespace for the column.
TRAILING_WHITESPACE:           Check for blank strings with unusual trailing whitespace for the column.
FIRST_CHAR_ALPHA:              Check if the first characters are consistently alphabetic within a column.
FIRST_CHAR_NUMERIC:            Check if the first characters are consistently numeric within a column.
FIRST_CHAR_SMALL_SET:          Check if there are a small number of distinct characters used for the first character
                                 within a column.
FIRST_CHAR_UPPERCASE:          Check if the first character is consistently uppercase within a column.
FIRST_CHAR_LOWERCASE:          Check if the first character is consistently lowercase within a column.
LAST_CHAR_SMALL_SET:           Check if there are a small number of distinct characters used for the last character
                                 within a column.
COMMON_SPECIAL_CHARS:          Check if there are one or more non-alphanumeric characters that consistently appear in the
                                 values within a column.
COMMON_CHARS:                  Check if there is consistently a small number of characters repeated in each value in a
                                 column.
NUMBER_ALPHA_CHARS:            Check if there is a consistent number of alphabetic characters in each value in a column.
NUMBER_NUMERIC_CHARS:          Check if there is a consistent number of numeric characters in each value in a column.
NUMBER_ALPHANUMERIC_CHARS:     Check if there is a consistent number of alphanumeric characters in  each value in a
                                 column.
NUMBER_NON-ALPHANUMERIC_CHARS: Check if there is a consistent number of non-alphanumeric characters in each value in a
                                 column.
NUMBER_CHARS:                  Check if there is a consistent number of characters in each value in a column.
MANY_CHARS:                    Check if any values have an unusually large number of characters for the column.
FEW_CHARS:                     Check if any values have an unusually small number of characters for the column.
POSITION_NON-ALPHANUMERIC:     Check if the positions of the non-alphanumeric characters is consistent within a column.
CHARS_PATTERN:                 Check if there is a consistent pattern of alphabetic, numeric and special characters in
                                 each value in a column.
UPPERCASE:                     Check if all alphabetic characters in a column are consistently uppercase.
LOWERCASE:                     Check if all alphabetic characters in a column are consistently lowercase.
CHARACTERS_USED:               Check if there is a consistent set of characters used in each value in a column.
FIRST_WORD_SMALL_SET:          Check if there is a small set of words consistently used for the first word of each value
                                 in a column.
LAST_WORD_SMALL_SET:           Check if there is a small set of words consistently used for the last word.of each value
                                 in a column.
NUMBER_WORDS:                  Check if there is a consistent number of words used in each value in a column.
LONGEST_WORDS:                 Check if a column contains any unusually long words.
COMMON_WORDS:                  Check if there is a consistent set of words used in each value in a column.
RARE_WORDS:                    Check if there are words which occur rarely in a given column.
GROUPED_STRINGS:               Check if a string or binary column is sorted into groups.
RARE_PAIRS:                    Check for pairs of values in two columns, where neither is rare, but the combination is
                                 rare.
RARE_PAIRS_FIRST_CHAR:         Check for pairs of values in two columns, where neither begins with a rare character, but
                                 the combination or first characters is rare.
RARE_PAIRS_FIRST_WORD:         Check for pairs of values in two columns, where neither begins with a rare word, but the
                                 combination of words is rare.
RARE_PAIRS_FIRST_WORD_VAL:     Check for pairs of values in two columns, where the combination of the first word in one
                                 and the value in the other is rare.
SIMILAR_CHARACTERS:            Check if two string columns, with one word each, consistently have a significant overlap
                                 in the characters used.
SIMILAR_NUM_CHARS:             Check if two string columns consistently have similar numbers of characters while the
                                 range of string lengths varies within both columns.
SIMILAR_WORDS:                 Check if two string columns consistently have a significant overlap in the words used.
SIMILAR_NUM_WORDS:             Check if two string columns consistently have similar numbers of words.
SAME_FIRST_CHARS:              Check if two string columns consistently start with the same set of  characters.
SAME_FIRST_WORD:               Check if two string columns consistently start with the same word.
SAME_LAST_WORD:                Check if two string columns consistently end with the same word.
SAME_ALPHA_CHARS:              Check if two string columns consistently contain the same set of  alphabetic characters.
SAME_NUMERIC_CHARS:            Check if two string columns consistently contain the same set of  numeric characters.
SAME_SPECIAL_CHARS:            Check if two string columns consistently contain the same set of special characters.
A_PREFIX_OF_B:                 Check if one column is consistently the prefix of another column.
A_SUFFIX_OF_B:                 Check if one column is consistently the suffix of another column.
B_CONTAINS_A:                  Check if one column is consistently contained in another columns, but is neither the
                                 prefix, nor suffix of the second column.
CORRELATED_ALPHA_ORDER:        Check if the alphabetic orderings of two columns are consistently correlated.
LARGE_GIVEN_VALUE:             Check if a value in a numeric column is very large given the value in a categorical
                                 column.
SMALL_GIVEN_VALUE:             Check if a value in a numeric column is very small given the value in a categorical
                                 column.
LARGE_GIVEN_PREFIX:            Check if a value in a numeric column is very large given the first word in a categorical
                                 column.
SMALL_GIVEN_PREFIX:            Check if a value in a numeric column is very small given the first word in a categorical
                                 column.
GROUPED_STRINGS_BY_NUMERIC:    Check if a string or binary column is sorted into groups when the table isordered by a
                                 numeric or date column.
LARGE_GIVEN_PAIR:              Check if a value in a numeric or date column is large given a pair of values in two string
                                 or binary columns.
SMALL_GIVEN_PAIR:              Check if a value in a numeric or date column is small given the a of values in two string
                                 or binary columns.
CORRELATED_GIVEN_VALUE:        Check if two numeric columns are correlated if conditioning on a string orbinary column.
DECISION_TREE_CLASSIFIER:      Check if a categorical column can be derived from the other columns using a decision tree.
C_IS_A_OR_B:                   Check if one column is consistently equal to the value in one of two other columns, though
                                 not consistently either one of the two columns.
TWO_PAIRS:                     Check that, given two pairs of columns, the first pair of columns have matching values in,
                                 and only in, the same rows as the other pair of columns.
UNIQUE_SETS_VALUES:            Check if a set of columns has consistently unique combinations of values.
MISSING_VALUES_PER_ROW:        Check if there is a consistent number of missing values per row.
ZERO_VALUES_PER_ROW:           Check if there is a consistent number of zero values per row.
UNIQUE_VALUES_PER_ROW:         Check if there is a consistent number of unique values per row.
NEGATIVE_VALUES_PER_ROW:       Check if there is a consistent number of negative values per row.
SMALL_AVG_RANK_PER_ROW:        Check if the numeric values in a row have a small average percentile value relative to
                                 their columns. This indicates the numeric values in a row are typically unusually small
                                 for their columns.
LARGE_AVG_RANK_PER_ROW:        Check if the numeric values in a row have a large average percentile value relative to
                                 their columns. This indicates the numeric values in a row are typically unusually large
                                 for their columns.
```
