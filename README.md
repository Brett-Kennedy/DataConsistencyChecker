# DataConsistencyChecker
A python tool to examine datasets for consistency. It performs approximately 150 tests, identifying patterns in the data and any exceptions to these. The tool provides useful analysis for any EDA (Exploratory Data Analysis) and may be useful for interpretable outlier detection. 

## Background

The central idea of the tool is to automate the tests that would be done by a person examining a dataset and trying to determine the patterns within the columns, between the columns, and between the rows (in cases where there is some meaning to the order of the rows). That is, the tool essentially does what a person, studying a new dataset, would do, but automatically. This does not remove the need to do manual data exploration and, where the goal is identifying outliers, other forms of outlier detection, but does cover much of the work identifying the patterns in the data and the exceptions to these, with the idea that these are two of the principle tasks to understand a dataset. The tool allows users to perform this work quicker, more exhaustively, and covering more tests than would normally be done. 

The tool executes a large set of tests over a dataset. For example, one test examines the number of decimal points typically found in a numeric column. If the data consistently contains, say, four decimal points, with no exceptions, this will be reported as a strong pattern without exceptions; if the data in this column nearly always has four decimal digits, with a small number of expections (for example having eight decimal digits), this will be reported as a strong pattern with exceptions. In this example, this suggests the identified rows may have been collected or processed in a different manner than the other rows. And, while this may not be interesting in itself, where rows are flagged multiple times for this or other issues, users can be more confident the rows are somehow different. 

The tests run over single columns (for example, checking for rare, unsually small or large values, etc), pairs of columns (for example checking if one column is usually larger than the other, contains similar characters (in the case of code or ID string values), etc.), or larger sets of columns (for example, checking if one column tends to be the sum, or mean of another set of columns). 

The unusual data found may be due to data collection errors, to mixing different types of data together, or other issues that may be considered errors, or that may be informative. 

### EDA

DataConsistencyChecker may be used for exploratory data analsys simply by running the tool and examining the patterns, and exceptions to the patterns that are found.

Exploratory data analysis may be run for a variety of purposes, but generally most relate to one of two high-level purposes: ensuring the data is of sufficient quality to build a model (or can be made to be of sufficient quality after removing or replacing any necessary values, rows, or columns), and gaining insights into the data in order to build appropriate models. This tool may be of assistence for both these purposes.  

### Outlier Detection

DataConsistencyChecker provides a large set of tests, each independent of the others and each higly interpretable. Running the tool, it's common for rows that are unusual to be flagged multiple times. This allows us to evaluate the outlier-ness of any given row based on the number of times it was flagged for issues relative to the other rows in the dataset. For example, if a row contains, for example, some values that are unusually large, as well as some strings with unusual characters, strings of unusual lengths, time values at unusual times of day, and rare values, the row will then be flagged multiple times, giving it a relatively high outlier score. 

The majority of outlier detectors work on either strictly numeric data, or strictly categorical data, with numeric outlier detectors seeking to identify unusual rows as points in high-dimensional space far from the majority of other rows, and with categorical outlier detectors seeking to identify unusual combinations of values. These tests are very useful, but can be uninterpretable with a sufficient number of features: it can be difficult to confirm the rows are more unusual than the majority of rows or to even determine why they were flagged as outliers. 

DataConsistencyChecker may flag a row, for example, for having a very large value in Column A, a value with an unsual number of digits in Column D, an unusually long string in Column C, and values with unusual rounding in Column D and Column G. As with any outlier detection scheme, it is difficult to gauge the outlier-ness of each of these, but DataConsistencyChecker does have the desirable property that each of these is simple to comprehend. 

As this tool is limited to interpretable outlier detection methods, it does not test for the multi-dimensional outliers detected by other algorithms such as Isolation Forest. These outlier detection algorithms should generally also be executed to have a full understanding of the outliers present in the data. 

## Intallation
The code consists of a single [python file](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/check_data_consistency.py) which may be downloaded and included in any projct. It does not rely on any other tools than numpy, pandas, scipy, matplotlib, seaborn, and other standard libraries. Once downloaded, it may be included as:

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

It is necessary to first instantiate a DataConsistencyChecker object, call init_data() with a pandas dataframe, and call check_data_quality(). After this is complete, there are a number of APIs available to examine the results and assess the findings, including APIs to summarize the patterns and exceptions by row, column, and test, to describe the most-flagged rows, and get the total score per row. In this example, display_detailed_results() is called, which displays each pattern and each exception found in detail, including examples of values for the relevant columns flagged and not flagged, and plots where appropriate. 

## Examples

In this section we provide some examples of patterns identified in datasets available on OpenML. See the example notebooks as well for examples of working with datasets from OpenML and the toy sets provided with sklearn. 

### Housing
In the sklearn housing dataset, the tool identified that AveRooms and AveOccup are consistently above 1.0, with 2 and 3 exceptions respectively. It also flags population values of 3, 6, and 8, and AveOccup values in the hundreds or thousands.

Note: in most cases, other patterns were also identified, which may or may not be interesting. There may be some step involved with examining the patterns to identify the relevant ones, but this is typically quite quick and worthwhile identify the interesting patterns. APIs are provided to assist with processessing patterns where many are discovered.

## Performance

The tool typically runs in under a minute for small files, and under 30 minutes even for very large files, with hundreds of thousands of rows and/or hundreds of columns. However, it is often useful to specify to execute the tool quicker, especially if calling it frequently, or with many datasets. Several techniques to speed the execution are listed here:

- Exclude some tests. Some may be of less interest to your project or may be slower to execute.  
- Set the max_combinations parameter to a lower value, which will skip a significant amount of processing in some cases. The default is 1,000,000. 
- Run on a sample of the rows. Doing this, you will not be able to properly identify exceptions to the patterns, but will be able to identify the patterns themselves. 
- Columns may be removed from the dataframe before running the tool to remove checks on these columns. 
- The fast_only parameters in check_data_quality() may be set to True
- Set the verbose level to 2 or 3. This allows users to monitor the progress better and determine if it is acceptable to wait, or if any of the other steps listed here should be tried.

## Date Columns

It is recommended that the pandas dataframe passed has any date columns set such that the dtype is datetime64 or timestamp. However, the tool will attempt to convert any columns with dtype as object to datetime, and will treat any that can be successfully converted as datetime columns, which means all tests related to date or time values will be executed on these columns. If verbose is set to 2 or higher, the column data types used will be displayed at the top ot the output running check_data_quality(). It is also possible to call display_columns_types_list() or display_columns_types_table() to check the datatypes that were inferred for the passed dataframe. If desired, you may pass a list of date columns, and the tool will attempt to treat all of these, and only these, columns as datetime columns. 

## Data Dredging and False Postives

This tool runs a large number of tests, some of which check many sets of columns. For example, the test to check if one column tends to be the sum of another set of columns, can check a very large number of subsets of the full set of numeric columns. This can lead to false positives, where patterns are reported that are random, or not meaningful. Ideally the output is clear enough that any patterns found may be confirmed, and checked to determine how relevant they are. A key property of the tool, though, is that each row is tested many times, so while there may be some false positives reported, rows flagged many times, particularly those reported many times for different tests on different columns, are certainly unusual. Note though, this says nothing regarding whether or not the rows are correct and useful, only that they are different from the other rows in the same dataset. 

### Internal Parameters

As with all outlier detectors, there are some thresholds and other decisions necessary to define what constitutes a pattern and what constitutes an exception to the pattern. There are options to configure the tests, such as the coefficients used for interquartile range and interdecile range tests. For the most part, though, we have taken the approach of trying to limit options, and to provide clear, sensible tests, without the need to experiment or run the tool multiple times. However, almost all tests could, at least potentially, be tweaked to run with different thresholds or with different options, which would produce different output. For example, one test exists to check if, for sets of numeric columns, if the values in these columns, row for row, tend to all be positive or all negative together. In this case, there is a decision how to handle zero values. While such decisions are made based on testing on a very large number of datasets, other decisions could be made. We have attempted in all cases to make the most reasonable decisions and, though alternative, similar tests may also be useful, the output here has been confirmed to be accurate and useful.

As another example, there are tests to determine if the values in one column can be predicted from the values in the other columns using a small decision tree or linear regression. In these cases, there are decisions related, for example, to the minimum accuracies of the models. As such, these tests should be taken simply as presenting the patterns it does find, without any statement about other patterns that could be found were different options used. Configuring these options does invite running through many combinations of options, increases data dredging and producing less combrehensible output. 

### Clearing Issues

All outlier detectors, including DataConsistencyChecker, allow some tuning. Detectors that return binary labels further require some thresholds, but all provide some options. For example, kth Nearest Neighbors requires defining k, a distance metric, a means of scaling each numeric column, a method of encoding categorical features, and so on. This is true of DataQualityChecker as well, though it has the advantage of executing many tests, such that any decisions made on individual tests will tend to average out over many tests, therefore requiring less tuning, though some tuning is possible if desired. Primarily though, we recommend tuning in the form of clearing issues, such that analysis begins with the full set of issues identified, followed by some manual pruning of any issues considered inconsequental, leaving the remaining issues, and a final outlier score per row. 

A set of APIs, clear_issue(), clear_issues_by_id(), and restore_issues() are provided to allow users to examine the results, and incrementally clear any issues found which may be considered non-issues, leaving a final set of patterns and exceptions that are consisidered noteworthy. This can help reduce some noise when a non-trivial number of issues are found, or when reporting issues to others. 

### Contamination Level

While exceptions to patterns necessarily require decisions related to thresholds, patterns without exceptions are more straight-forward. In most cases, the pattern clearly exists or does not. For example a numeric column is monotonically increasing or it is not; a string column consistently contains 4 uppercase characters or it does not. These are typically clear, and can be quite straightforward to assess. With exceptions, however, some decisions must be made about the nature of the exceptions, but in most cases, this is simply a case of specifiying the number of exceptions permitted such that we recognize a pattern as still existing, albeit with exceptions. 

In general, much of tuning outlier detectors in determining how many outliers should be flagged. With DataConsistencyChecker, this is done most simply with the contamination_level parameter, which defines the maximum number or exceptions to allow to a pattern for it to be considered a pattern with exceptions, as opposed to a non-pattern. This may be specified as an absolute count or as a fraction of the total number of rows. The default is set to 0.5%, which we have found quite sensible, but tuning this may be quite reasonable as well. 

If there are, say, 10,000 rows in a dataset, then the default value of 0.5% exceptions to any pattern allows here 50 rows. For example, one test checks if two columns have values such that one is the result of rounding the other. In this example, if this is true in all 10,000 rows, this will be identified as a pattern with no exceptions; if this is true in all but 1 to 50 rows, this will be considered a pattern with exceptions, and if false in 51 or more rows, it will not be considered a pattern. Adjusting the contamination_level up or down can provide a more suitable number of exceptions for some cases. 

### Noting the Patterns not Found

DataConsistencyChecker performs enough tests that it's reasonably likely to find something of interest in any non-trivial dataset, and there is some meaning in the fact that it does not when it does not. In general, what it does not flag can be of interest as well as what it does flag. For example, any rows with scores of zero may be reasonably considered to be very normal with respect to the dataset. As well, the tool provides interfaces to identify where patterns were not present. For example, if it may be expected that a column contains entirely positive values, then it is possible to see that it is not the case that it contains entirely positive values, or that it contains almost entirely positive values, with some exceptions.  

## Synthetic Data
The tool provides an API to generate synthetic data, which has been designed to provide columns that will be flagged by the tests, such that each test flags at least one column or set of columns. This may be useful for new users to help understand the tests, as they provide useful examples, and may be altered to determine when specifically the tests flag or do not flag patterns. The synthetic data is also used by the unit tests to help ensure consistent behaviour as the tests are expanded and improved. 

## Unit Tests
The unit tests are organized such that each test .py file tests one DataQualityChecker test, both on real and synthetic data. 

### Unit Tests on Synthetic data with null values
As null values are common in real-world data, each test must be able to handle null values in a defined manner. For most tests, the null values do not support or violate the patterns discovered. 

### Synthetic data on all columns vs those specific to the tests
When generating synthetic data, a set of columns, typically two to five columns, will be created in order to exercise the related test. Each set of columns is relevant to only one test, though, it created, will increase the test coverage of the other tests. For example, to test FEW_WITHIN_RANGE, the synthetic data created three columns: 'few in range rand', 'few in range all', and ''few in range most'. 

It is possible to call check_data_quality() on either a specific set of tests, or without specifying a set of tests, which will result in all tests being executed. Where only a subset of the available tests are executed, it may be desired to generate only the columns relevant to these tests. This will result in faster execution times and more predictable results. It is also possilble to specify to create all synthetic columns, including those designed for other tests. These columns will be covered as well by any tests executed and will provide more testing. Typically any columns flagged by a test will be flagged regardless if other synthetic columns are generated or not. 
