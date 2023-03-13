# DataConsistencyChecker
A python tool to examine datasets for consistency. Performs approximately 130 tests. A useful step for any EDA (Exploratory Data Analysis) and for interpretable outlier detection.

## Background

This tool is actually a tool for EDA, and an interpretable outlier detector. It executes a large set of tests over a dataset, looking both for strong patterns and exceptions to any strong patterns found. For example, one test examines the number of decimal points typically found in a numeric column. If the data consistently contains four decimal points, with no exceptions, this will be reported as a strong pattern without exceptions; if the data in this column almost always has four decimal points, with a small number of expections, this will be reported as a strong pattern with exceptions. 

Many tests run over single columns (for example, checking for rare, unsually small or large values, etc), pairs of columns (for example checking if one column is usually larger than the other, contains similar characters (in the case of code or ID string values), etc.), or larger sets of columns (for example, checking if one column tends to be the sum or mean of another set of columns). 

As the tests check a large number of combinations, some things will be flagged that are simply random quirks of the data. However, the rows and the columns that are flagged many times, particularly for many different types of issues, can safely be said to be unusual. 

The unusual data may be due to data collection errors, to mixing different types of data together, or other issues that may be considered errors, or that may be informative. 

### EDA

DataConsistencyChecker may be used for exploratory data analsys simply by running the tool and examining the patterns, and exceptions to the patterns that are found.

Exploratory data analysis may be run for a variety of purposes, but generally most relate to one of two high-level purposes: ensuring the data is of sufficient quality to build a model (or can be made to be of sufficient quality after removing or replacing any necessary values, rows, or columns), and gaining insights into the data in order to build appropriate models. 

### Outlier Detection

DataConsistencyChecker provides a large set of tests, each independent of the others and each higly interpretable. Given that, it's common for rows that are unusual to be flagged multiple times. This allows us to evaluate the outlier-ness of any given row based on the number of times it was flagged for issues relative to the other rows in the dataset. 

The majority of outlier detectors work on either strictly numeric data, or strictly categorical data, with numeric outlier detectors seeking to identify unusual rows as points in high-dimensional space far from the majority of other rows, and with categorical outlier detectors seeking to identify unusual combinations of values. These tests are very useful, but can be uninterpretable with a sufficient number of features: it can be difficult to confirm the rows are more unusual than the majority of rows or to determine even why they were flagged as outliers. 

DataConsistencyChecker may flag a row, for example, for having a very large value in Column A, a value with an unsual number of digits in Column D, an unusually long string in Column C, and values with unusual rounding in Column D and Column G. As with any outlier detection scheme, it is difficult to gauge the outlier-ness of each of these, but DataConsistencyChecker does have the desirable property that each of these is simple to comprehend. 


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
