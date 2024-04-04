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

A major drawback of most outlier detectors is they can be very much blackboxes. The detectors are very useful, but can be uninterpretable, especially with high-dimensional datasets: it can be difficult to confirm the rows are more unusual than the majority of rows or to even determine why they were flagged as outliers. DataConsistencyChecker's tests are each interpretable, and the final scoring system is completely transparent. 

Running the tool, it's common for rows that are unusual to be flagged multiple times. This allows users to evaluate the outlierness of any given row based on the number of times it was flagged for issues, relative to the other rows in the dataset. For example, if a row contains some values that are unusually large, as well as some strings with unusual characters, strings of unusual lengths, time values at unusual times of day, and rare values in categorical columns, the row will then be flagged multiple times, giving it a relatively high outlier score. 

As this tool is limited to interpretable outlier detection methods, it does not test for the multi-dimensional outliers detected by other algorithms such as Isolation Forest, Local Outlier Factor, Angle-based Outlier Detection, Cluster-based outlier detection, etc. These outlier detection algorithms should generally also be executed to have a full understanding of the outliers present in the data. 

The unusual data found may be due to data collection errors, mixing different types of data together, or other issues that may be considered errors, or that may be informative. Some may point to forms of feature engineering, which may be useful for downstream tasks. 


## Intallation
The code consists of a single [python file](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/check_data_consistency.py) which may be downloaded and included in any project. It uses one package, which must be installed as:

pip intall termcolor

Otherwise, it does not rely on any packages other than numpy, pandas, scipy, matplotlib, seaborn, and other standard libraries. Once downloaded, it may be included as:

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

## Example with More Output
```python
import pandas as pd
from check_data_consistency import DataConsistencyChecker

dc = DataConsistencyChecker()
dc.init_data(df)
dc.check_data_quality()
dc.summarize_patterns_and_exceptions()
dc.display_detailed_results(test_id_list=['LARGE_GIVEN_DATE'])
```

Where many patterns, with or without exceptions, are found, it may be impractical to simply call display_detailed_results() to view detailed descriptions of each pattern found. In these cases, it is possible to call an API such as summarize_patterns_and_exceptions() (there are several such APIs to list or summarize the findings) first to get an overview of what was found. In some cases, this overview may be all that is necessary, without looking at the findings in detail. However, where users wish to drill down further, the display_detailed_results() API may be called specifying a set of tests, columns, row ids, pattern ids, or exceptions ids, which allows you to focus on those findings interesting for your data and goals. In this example, we assume LARGE_GIVEN_DATE was one of the tests identified by summarize_patterns_and_exceptions(), and is of interest to the user. 

## 2nd Example with More Output
```python
import pandas as pd
from check_data_consistency import DataConsistencyChecker

dc = DataConsistencyChecker()
dc.init_data(df)
dc.check_data_quality()
dc.summarize_patterns_and_exceptions()
dc.display_next()
```
The display_next() API will output the results (or a sample of the results if there are many) for a single test. If this is called repeatedly, it will, on each execution, display the results for the next test for which there are results, ordering the tests in the same order in which they are executed. Where it is desirable to save the results in a notebook, a new cell may be used for the next call to display_next(). Where the results may be over-written (and this is preferred where there are many results to avoid memory issues), display_next() may be called repeatedly in the same cell, presenting the results for the next test each time. 


## 3rd Example with More Output
```python
import pandas as pd
from check_data_consistency import DataConsistencyChecker

dc = DataConsistencyChecker()
dc.init_data(df)
dc.check_data_quality()
dc.summarize_patterns_and_exceptions()
dc.display_detailed_results(save_to_disk=True)
```
Saving to disk will create an HTML file called Data_Consistency.html with the full results. The folder may be specified for this. This can contain a large volume of output as it does not need to be rendered within a notebook.


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


**Test_Demo API Demo**

The [Test_Demo API Demo](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_Test_Method.ipynb) notebook demonstrates the test_demo() API, which provides examples of a specified test ID using the provided synthetic data. 

## Full API
For a description of the APIs, see: [Full API Documentation](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/docs/api.md)

## Performance
For note on reducing the execution times of the analysis, refer to:
[Performance Notes](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/docs/performance.md)


## Unit Tests
For contributors: [Notes on unit tests](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/docs/unit_tests.md)

## Additional Documenation
Notes on additional topics, inlcuding date columns, clearing issues, contamination levels, the sort order of the data, and the use of synthetic may be found at: [Additional Documentation](https://github.com/Brett-Kennedy/DataConsistencyChecker/blob/main/docs/additional_documentation.md)

