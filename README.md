# DataConsistencyChecker
A python tool to examine datasets for consistency. It performs approximately 140 tests, identifying patterns in the data and any exceptions to these. The tool provides useful analysis for any EDA (Exploratory Data Analysis) and may be useful for interpretable outlier detection.

## Background

The central idea of the tool is to automate the tests that would be done by a person examining a dataset and trying to determine the patterns within the columns, between the columns, and between the rows (in cases where there is some meaning to the order of the rows). That is, the tool essentially does what a person, studying a new dataset, would do, but automatically. 

The tool executes a large set of tests over a dataset, looking both for strong patterns and exceptions to any strong patterns found. For example, one test examines the number of decimal points typically found in a numeric column. If the data consistently contains, say, four decimal points, with no exceptions, this will be reported as a strong pattern without exceptions; if the data in this column almost always has four decimal digits, with a small number of expections (for example having eight decimal digits), this will be reported as a strong pattern with exceptions. 

The tests run over single columns (for example, checking for rare, unsually small or large values, etc), pairs of columns (for example checking if one column is usually larger than the other, contains similar characters (in the case of code or ID string values), etc.), or larger sets of columns (for example, checking if one column tends to be the sum or mean of another set of columns). 

The unusual data found may be due to data collection errors, to mixing different types of data together, or other issues that may be considered errors, or that may be informative. 

### EDA

DataConsistencyChecker may be used for exploratory data analsys simply by running the tool and examining the patterns, and exceptions to the patterns that are found.

Exploratory data analysis may be run for a variety of purposes, but generally most relate to one of two high-level purposes: ensuring the data is of sufficient quality to build a model (or can be made to be of sufficient quality after removing or replacing any necessary values, rows, or columns), and gaining insights into the data in order to build appropriate models. 

### Outlier Detection

DataConsistencyChecker provides a large set of tests, each independent of the others and each higly interpretable. Running the tool, it's common for rows that are unusual to be flagged multiple times. This allows us to evaluate the outlier-ness of any given row based on the number of times it was flagged for issues relative to the other rows in the dataset. For example, if a row contains, for example, some values that are unusually large, as well as some strings with unusual characters, strings of unusual lengths, time values at unusual times of day, and rare values, then the row will be flagged multiple times, giving it a relatively high outlier score. 

The majority of outlier detectors work on either strictly numeric data, or strictly categorical data, with numeric outlier detectors seeking to identify unusual rows as points in high-dimensional space far from the majority of other rows, and with categorical outlier detectors seeking to identify unusual combinations of values. These tests are very useful, but can be uninterpretable with a sufficient number of features: it can be difficult to confirm the rows are more unusual than the majority of rows or to determine even why they were flagged as outliers. 

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
dc.check_data_quality(df)
dc.display_detailed_results()
```

## Documentation

## Performance

The tool typically runs in under a minute for small files, and under 30 minutes even for very large files, with many thousands of rows and/or columns. However, it is
often useful to specify to execute the tool quicker, especially if calling it frequently, or with many datasets. Several techniques to speed the execution are listed here:

- Excluding some tests. Some may be of less interest to your project or may be slower to execute.  
- Set the max_combinations parameter to a lower value, which will skip a significant amount of processing in some cases. The default is 1,000,000. 
- Run on a sample of the rows. Doing this, you will not be able to properly identify exceptions to the patterns, but will be able to identify the patterns themselves. 
- Columns may be removed from the dataframe before running the tool to remove checks on these columns. 
- The fast_only parameters in check_data_quality() may be set to True

## Date Columns

It is recommended that the pandas dataframe passed has any date columns set such that the dtype is datetime64 or timestamp. However, the tool will attempt to convert any columns with dtype as object to datetime, and will treat any that can be successfully converted as datetime columns, which means all tests related to date or time values will be executed on these columns. If verbose is set to 2 or higher, the column data types used will be displayed at the top ot the output running check_data_quality(). If desired, you may pass a list of date columns, and the tool will attempt to treat only these columns as datetime columns. 


## Notes

(create separate sections for these)

This tool runs a large number of tests, some of which check many sets of columns. For example, the test to check if one column tends to be the sum of another set of columns, can check a very large number of subsets of the full set of numeric columns. This can lead to false positives, where patterns are reported that are random, or not meaningful. Ideally the output is clear enough that any patterns found may be confirmed, and checked to determine how relevant they are. A key property of the tool, though, is that each row is tested many times, so while there may be some false positives reported, rows flagged many times, particularly those reported many times for different tests on different columns, are certainly unusual. Note though, this says nothing regarding whether or not the rows are correct and useful, only that they are different from the other rows in the same dataset. 

There are options to configure some parameters, such as the coefficients used for interquartile range and interdecile range tests. For the most part, we have taken the approach of trying to limit options, and to provide clear, sensible tests. However, almost all tests could be tweaked to run with different thresholds or with different options, which would produce different output. For example, there is a test to check if, for sets of numeric columns, if the values in these columns, row for row, tend to all be positive or all negative together. In this case, there is a decision how to handle zero values. There are tests to see if the values in one column can be predicted from the values in the other columns using a small decision tree or a linear regression. In these cases, there are decisions related to the minimum accuracies of the models. As such, these tests should be taken simply as presenting the patterns it does find, without any statement about other patterns that could be found were different options used. Configuring these options does invite running through many combinations of options, increases data dredging and producing less combrehensible output. 

While exceptions to patterns necessarily require decisions related to thresholds, paterns without exceptions are more straight-forward. In most cases, the pattern clearly exists or does not. For example a numeric column is monotonically increasing or it is not. With exceptions, some decisions must be made about the nature of the exceptions, but in most cases, this is simply a case of specifiying the number of exceptions permitted such that we recognize a pattern as still existing, albeit with exceptions. This is configurable, though with, we believe, a sensible default of 0.5%, used equally for all tests. 
