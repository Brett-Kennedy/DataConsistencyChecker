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
**DataConsistencyChecker**(iqr_limit=3.5, idr_limit=1.0, max_combinations=100_000, verbose=1)

        Initialize a DataConsistencyChecker object.

        iqr_limit: float
            Inter-quartile range is used in several tests to find unusually small or large values. For example,
            to identify large values, Q3 (the 3rd quartile, or 75th percentile plus some multiplier times the
            inter-quartile range is often used. Ex: Q3 + 1.5*IQR. To avoid noisy results, a higher coefficient
            is used here.

        idr_limit: float
            Inter-decile range is used in several tests, particularly to identify very small values, as IQR can work
            poorly where the values are strictly positive.

        max_combinations: int
            Several tests check many combinations of columns, which can be very slow to execute where there are many
            columns. For example, BIN_NUM_SAME will check subsets of the binary columns, testing different sizes of
            subsets. For some sizes of subsets, there may be a very large number of combinations. Setting
            max_combinations will restrict the number of combinations examined. This may result in missing some
            patterns, but allows execution time to be limited. This applies only to tests that check subsets of
            multiple sizes.

        verbose: int
           -1: no output at all will be displayed
            0: no output will be displayed until the tests are complete.
            1: the test names will be displayed as they execute.
            2: a description of each test, and progress related to each of the more expensive tests will be displayed
               as well.

**init_data**(df, known_date_cols=None)

        df: dataframe to be assessed

        known_date_cols: list of strings
            If specified, these, and only these, columns will be treated as date columns.

**check_data_quality**(append_results=False, execute_list=None, exclude_list=None, test_start_id=0, fast_only=False,
                    include_code_tests=True, freq_contamination_level=0.005, rare_contamination_level=0.1,run_parallel=False)

        Run the specified tests on the dataset specified init_data(). This method identifies the patterns and exceptions to
        these found in the data. Additional API calls may be made to access the results.

        append_results: bool
            If set True, any previous test results, from previous executions of check_data_quality() will be saved and
            the results from the current run appended to the previous results. If False, all previous test results will
            be removed.

        execute_list: list of Test IDs
            If specified, these and only these tests will be executed.

        exclude_list:  list of Test IDs
            If specified, all tests other than these will be executed. It is not permitted to specify both
               execute_list and exclude_list.

        test_start_id: int
            Each test has a unique number. Specifying a value greater than 0 will skip the initial tests. This may be
            specified to continue a previous execution that was incomplete.

        fast_only: bool
            If specified, only tests that operate on single columns will be executed. The slower tests check sets of
            two or more columns, and are skipped if this is set True.

        include_code_tests: bool:
            Some tests are specific to columns with code or ID values, such that the individual characters in the
            values may have meaning. For example, with value X7333, it may be relevant that the first character is
            an 'X', or that the subsequent characters are 4 numeric characters. If set True, these tests will be
            executed.

        freq_contamination_level: int or float
            The maximum fraction of rows in violation of the pattern where we consider the pattern to still be in place.
            If set as an integer, this defines the maximum number of rows, as opposed to the fraction. This is used for 
            tests that frequently find results and is set low to reduce over-reporting

        rare_contamination_level: int or float
            This is used for tests that rarely find results and is set high to reduce under-reporting

        run_parallel: bool
            If set True, the tests will be run in parallel, which can reduce overall execution time.


### Methods to generate or modify test data

**generate_synth_data**(all_cols=False, execute_list=None, exclude_list=None, seed=0, add_nones="none")

        Generate a random synthetic dataset which may be used to demonstrate each of the tests.

        all_cols: bool
            If all_cols is False, this generates columns specifically only for the tests that are specified to run.
            If all_cols is True, this generates columns related to all tests, even where that test is not specified to
            run.

        execute_list: list of strings
            If specified, only columns related to these tests will be generated.

        exclude_list: list of strings
            If specified, columns related to all tests other than these will be generated. It is not permitted to
            specify both execute_list and exclude_list.

        seed: int
            If specified, this is used to initialize any random processes, which are involved in the creation of
            most synthetic columns. If specified, the synthetic data creation will be repeatable.

        add_nones: string
            Must be one of 'none', 'one-row', 'in-sync', 'random', '80-percent'.
            If add_nones is set to 'random', then each column will have a set of None values added randomly, covering
            50% of the values. If set to 'in-sync', this is similar, but all columns will have None set in the ame rows.
            If set to 'one-row', only one row will be given None values. If set to '80-percent', 80% of all values
            will be set to None.

**modify_real_data**(df, num_modifications=5)

        Given a real dataset, modify it slightly, in order to add what are likely inconsistencies to the data

        df: pandas dataframe
            A real or synthetic dataset.

        num_modifications: int
            The number of modifications to make. This should be small, so as not to change the overall distribution
            of the data

        Return:
            the modified dataframe,
            a list of row numbers and column names, indicating the cells that were modified


### Methods to output information about the tool, unrelated to any specific dataset or test execution

**get_test_list**():
        
        Returns a python list, listing each test available by ID.
        
**get_test_descriptions**()
        
        Returns a python dictionary, with each test ID as key, matched with a short text explanation of the test.
        
        
**print_test_descriptions**(long_desc=False)
        
        Prints to screen a prettified list of tests and their descriptions.

        long_desc: bool
            If True, longer descriptions will be displayed for tests where available. If False, the short descriptions
            only will be displayed.
        

  **get_patterns_shortlist**()
        
        Returns an array with the IDs of the tests in the short list. These are the tests that will be presented by
        default when calling get_patterns() to list the patterns discovered.
        

**get_tests_for_codes**()
        
        Returns an array with the IDs of the tests that related to ID and code values. Where it is known that no columns
        are of this type, these tests may be skipped.
        


**demo_test**(test_id, include_nulls=False)
        
        This provides a demo of a single test.

        This creates and displays synthetic demo data, runs the specified test on the demo data, and outputs the results, 
        calling display_detailed_results().

        For most tests, the associated synthetic test data has three to five columns. The columns are named in two parts 
        with the first indicating the test the synthetic data is designed to test and demonstrate, and the other part 
        indicating if the column is part of a pattern with or without an exception. Typically those named 'rand' or 
        'rand_XXX' do not have a pattern, but may be involved in patterns spanning multiple columns. Those named 'all' 
        are part of a pattern with no exceptions (and sometimes other patterns as well). Those named 'most' are part of 
        patterns with exceptions, indicating the pattern holds for most, but not all, rows.

        include_nulls (bool):
            If True, several versions of the data will be created with either few or many Null values.
        

### Methods to output statistics about the dataset, unrelated to any tests executed.

**display_columns_types_list**()
        
        Displays, for each of the four column types identified by the tool, which columns in the data are of those
        types. This may be called to check the column types were identified correctly. This will skip columns removed
        from analysis due to having only one unique value.
        

**display_columns_types_table**()
        
        Displays the first rows of the data, along with the identified column type in the first row. Similar to
        calling display_columns_types_list(), this may be used to determine if the inferred column types are correct.
        
        
### Methods to output the results of the analysis in various ways

**get_test_ids_with_results**(include_patterns=True, include_exceptions=True)
        
        Gets a list of test ids, which may be used, for example, to loop through tests calling other APIs such as
        display_detailed_results()

        include_patterns: bool
            If True, the returned list will include all test ids for tests that flagged at least one pattern without
            exceptions

        include_exceptions: bool
            If True, the returned list will include all test ids for tests that flagged at least one pattern with
            exceptions

        Returns: array of test ids
            Returns an array of test ids, where each test found at least one pattern, with or without exceptions, as
            specified
        
        
**get_single_feature_tests_matrix**():
        
        Returns a matrix with a column for every column in the original data and a row for each test executed. The cell
        values contain the percent of rows matching the pattern associated with the test. In many cases, this will be
        NaN (rendered as '-'), as not all tests execute on all columns. For example, tests for large numeric values
        execute only on numeric columns. As well, for efficiency, many tests skip examining columns that have many null
        values, many zero values, few or many unique values, etc. This in necessary to make the execution on many tests
        tractable where there are many columns and/or many rows.
        

**get_patterns_list**(test_exclude_list=None, column_exclude_list=None, show_short_list_only=True)
        
        This returns a dataframe containing a list of all, or some, of the identified patterns that had no exceptions.
        Which patterns are included is controlled by the parameters. Each row of the returned dataframe represents
        one pattern, which is one test over some set of rows. The dataframe specifies for each pattern: the test,
        the set of columns, and a description of the pattern.

        test_exclude_list: list
            If set, rows related to these tests will be excluded.

        column_exclude_list: list
            If set, rows related to these columns will be excluded.

        show_short_list_only: bool
            If True, only the tests that are most relevant (least noisy) will be returned. If False, all identified
            patterns matching the other parameters will be returned.
        
**get_exceptions_list**()
        
        Returns a dataframe containing a row for each pattern that was discovered with exceptions. This has a similar
        format to the dataframe returned by get_patterns_list(), with one additional column representing the number
        of exceptions found. The dataframe has columns for: test id, the set of columns involved in the pattern,
        a description of the pattern and exceptions, and the number of exceptions.
        

**get_exceptions**()
        
        Returns a dataframe with the same set of rows as the original dataframe, but a column for each pattern that was
        discovered that had exceptions, and a column indicating the final score for each row. This dataframe can be very 
        large and is not generally useful to display, but may be collected for further analysis.

        Returns:
            pandas.DataFrame: DataFrame containing exceptions and final scores
        

**get_exceptions_by_column**()
        
        Returns a dataframe with the same shape as the original dataframe, but with each cell containing, instead
        of the original value for each feature for each row, a score allocated to that cell. Each pattern with
        exceptions has a score of 1.0, but patterns the cover multiple columns will give each cell a fraction of this.
        For example with a pattern covering 4 columns, any cells that are flagged will receive a score of 0.25 for
        this pattern. Each cell will have the sum of all patterns with exceptions where they are flagged.
        
        
**summarize_patterns_by_test_and_feature**(all_tests=False, heatmap=False)
        
        Create and return a dataframe with a row for each test and a column for each feature in the original data. Each
        cell has a 0 or 1, indicating if the pattern was found without exceptions in that feature. Note, some tests to
        not identify patterns, such as VERY_LARGE. The dataframe is returned, and optionally displayed as a heatmap.

        all_tests: bool
            If all_tests is True, all tests are included in the output, even those that found no patterns. This may be
            used specifically to check which found no patterns.

        heatmap: bool
            If True, a heatmap will be displayed
        

**summarize_exceptions_by_test_and_feature**(all_tests=False, heatmap=False)
        
        Create a dataframe with a row for each test and a column for each feature in the original data. Each cell has an
        integer, indicating, if the pattern was found in that feature, the number of rows that were flagged. Note,
        at most contamination_level of the rows (0.5% by default) may be flagged for any test in any feature, as this
        checks for exceptions to well-established patterns.

        all_tests: bool
            If all_tests is True, all tests are included in the output, even those that found no issues. This may be
            used specifically to check which found no issues.

        heatmap: bool:
            If set True, a heatmap of the dataframe will be displayed.
        

**summarize_patterns_by_test**(heatmap=False)
        
        Create and return a dataframe with a row for each test, indicating 1) the number of features where the pattern
        was found.

        heatmap: bool
            If True, a heatmap of the results are displayed
        
**summarize_exceptions_by_test**(heatmap=False)
        
        Create and return a dataframe with a row for each test, indicating 1) the number of features where the pattern
        was found but also exceptions, 2) the number of issues total flagged (across all features and all rows).

        heatmap: bool
            If True, a heatmap of the results are displayed
                
**summarize_patterns_and_exceptions**(all_tests=False, heatmap=False)
        
        Returns a dataframe with a row per test, indicating the number of patterns with and without exceptions that
        were found for each test. This may be used to quickly determine which pattens exist within the data, and
        to help focus specific calls to display_detailed_results() to return detailed information for specific tests.

        all_tests: bool
            If True, a row will be included for all tests that executed. This will include tests where no patterns were
            found. If False, a row will be included for all tests that found at least one pattern, with or without
            exceptions.

        heatmap: bool
            If True, a heatmap of form of the table will be displayed.

**display_detailed_results**(
            test_id_list=None,
            col_name_list=None,
            issue_id_list=None,
            pattern_id_list=None,
            row_id_list=None,
            show_patterns=True,
            show_exceptions=True,
            show_short_list_only=True,
            include_examples=True,
            plot_results=True,
            max_shown=-1
            )
        
        Loops through each test specified, and each feature specified, and presents a detailed description of each. If
        filters are not specified, the set of identified patterns, with and without exceptions, can be very long in
        some cases, and in these cases, the method will not be able to display them. In this case, additional filters
        should be specified.

        test_id_list: Array of test IDs
            If specified, only these will be displayed. If None, all tests for which there is information to be display
            will be displayed.

        col_name_list: Array of column names, matching the column names in the passed dataframe.
            If specified, only these will be displayed, though the display will include any patterns or exceptions that
            include these columns, regardless of the other columns. If None, all columns for which there is information
            to display will be displayed.

        issue_id_list: Array of Issue IDs
            If specified, only these exceptions will be displayed. If set, patterns will not be shown.

        pattern_id_list: Array of Pattern IDs
            If specified, only these patterns will be displayed. If set, exceptions will not be shown.

        row_id_list: Array of ints, representing row numbers in the original dataset
            If specified, only patterns or exceptions found for these rows will be displayed.

        show_patterns: bool
            If set True, patterns without exceptions will be displayed. If set False, these will not be displayed.

        show_exceptions: bool
            If set True, patterns with exceptions will be displayed. If set False, these will not be displayed.

        show_short_list_only: bool.
            If False, all identified patterns matching the other parameters will be returned. If True, only the tests
            that are most relevant (least noisy) will be displayed as patterns. This does not affect the exceptions
            displayed.

        include_examples: bool
            If True, for any patterns found, examples of a random set of rows (other than cases where row order is
            relevant, in which case a consecutive set of rows will be used), with the relevant set of columns, will
            be display. As well, for any patterns found with exceptions, both a random set of rows that are not flagged
            and that are flagged will be displayed, also with only the relevant columns. May be set False to save
            time and space displaying the results.

        plot_results: bool
            If True, for any tests where plots are possible, one or more plots will be shown displaying the patterns.
            If exceptions are found, they will typically be shown in red. May be  set False to save
            time and space displaying the results.

        max_shown: int
            The maximum total number of patterns and exceptions shown. If no filters are set, the function will return
            if more patterns and/or exceptions are available. If filters are set, and the number is larger, the first
            max_shown will be set. If set to -1, a default will be used, which considers if plots and examples are
            to be displayed. The default is 200 without plots or examples, 100 with either, and 50 with both.
        

**display_next**()
        
        This may be used where there are many results, and we wish to view detailed descriptions of all or most of
        these. This API calls display_detailed_results() for one test at a time, for each test that identified at least
        one pattern (with or without exceptions) during the last call to check_data_quality(). This allows, when working
        with notebooks, for output to be spread over multiple cells, which can make viewing it simpler. Note though,
        where many tests flag patterns, in most cases only a subset of these would be useful to examine in detail,
        though this varies for different projects.
        

**get_outlier_scores**()
        
        Returns an outlier score for each row, similar to most outlier detectors.
        Returns a python array with an element for each row in the original data. All values are non-negative integer
        values, with most rows containing zero for most datasets.
        
**get_results_by_row_id**(row_num)
        
        Returns a list of tuples, with each tuple containing a test ID, and column name, for all issues flagged in the
        specified row.
        

**plot_final_scores_distribution_by_row**()
        
        Display a probability plot and histogram representing the distribution of final scores by row.
        

**plot_final_scores_distribution_by_feature**()
        
        Display a bar plot representing the distribution of final scores by feature.
        

 **plot_final_scores_distribution_by_test**()
        
        Display a bar plot representing the distribution of final scores by test.
               

**display_least_flagged_rows**(with_results=True, n_rows=10)
        
        This displays the n_rows rows from the original data with the lowest scores. These are the rows with the least
        flagged issues. This may be called to provide context for the flagged rows. In rare cases, some returned rows
        may have non-zero scores, if all or most rows in the dataset are flagged at least once.

        with_results: bool
            If with_results is False, this displays a single dataframe showing the appropriate subset of the original
            data. If with_results is True, this displays a dataframe per original row, up to n_rows. For each, the
            original data is shown, along with all flagged issues, across all tests on all features.

        n_rows: int
            The maximum number of original rows to present.
        

**display_most_flagged_rows**(with_results=True, n_rows=10)
        
        This is similar to display_least_flagged_rows, but displays the rows with the most identified issues.

        with_results: bool
            If True, the flagged rows will be display in separate tables, with additional rows indicating which tests
            flagged which columns. If False, all displayed rows will be displayed in a single table; the flagged
            columns will be highlighted, but there will not be an indication of which tests flagged them.
        n_rows: int
            The maximum number of original rows to present.
        
**quick_report**()
        
        A convenience method, which calls several other APIs, to give an overview of the results in a single API.
        

### Methods to find relationships between the data and the numbers of issues found

plot_columns_vs_final_scores()
        
        Used to determine if there are any relationships between column values and the final scores of the rows. This
        displays tables and plots presenting any relationships found.
        


## Performance
The tool typically runs in under a minute for small files and under 30 minutes even for very large files, with hundreds of thousands of rows and hundreds of columns. However, it is often useful to specify to execute the tool quicker, especially if calling it frequently, or with many datasets. Several techniques to speed the execution are listed here:

- Exclude some tests. Some may be of less interest to your project or may be slower to execute.  
- Set the max_combinations parameter to a lower value, which will skip a significant amount of processing in some cases. The default is 100,000. The value relates to the number of sets of columns examined. For example, if the test checks each triple of columns, such as with SIMILAR_TO_RATIO, then the number of sets of three numeric columns is the number of combinations. This is equal to (N choose 3), where N is the number of numeric columns. With verbose set to 2 or higher, it's possible to see the number of combinations required and the current setting of max_combinations. 
- Run on a sample of the rows. Doing this, you will not be able to properly identify exceptions to the patterns, but will be able to identify the patterns themselves. 
- Columns may be removed from the dataframe before running the tool to remove checks on these columns. 
- The fast_only parameters in check_data_quality() may be set to True. Setting this, only tests that are consistently fast will be executed.
- Set include_code_tests to False. Where it is known that none of the columns represent ID or Code values, the tests associated with these types of columns may be skipped. These tests tend to be fast to execute in any case, but removing these can save some time as well as noise from the results. 
- Set the verbose level to 2 or 3. This allows users to monitor the progress better and determine if it is acceptable to wait, or if any of the other steps listed here should be tried.

## Date Columns

It is recommended that the pandas dataframe passed has any date columns set such that the dtype is datetime64 or timestamp. However, the tool will attempt to convert any columns with dtype as object to datetime, and will treat any that can be successfully converted as datetime columns, which means all tests related to date or time values will be executed on these columns. If verbose is set to 2 or higher, the column data types used will be displayed at the top ot the output running check_data_quality(). It is also possible to call display_columns_types_list() or display_columns_types_table() to check the data types that were inferred for the passed dataframe. If desired, you may pass a list of date columns to init_data() and the tool will attempt to treat all of these, and only these, columns as datetime columns. 

## Data Dredging and False Postives

This tool runs a large number of tests, some of which check many sets of columns. For example, the test to check if one column tends to be the sum of another set of columns, can check a very large number of subsets of the full set of numeric columns. This can lead to false positives, where patterns are reported that are random, or not meaningful. Ideally the output is clear enough that any patterns found may be confirmed, and checked to determine how relevant they are. A key property of the tool, though, is that each row is tested many times, and so while there may be some false positives reported, rows flagged many times, particularly those reported many times for different tests on different columns, are certainly unusual. Note though, reliably identifying rows as unusual does not suggest anything regarding whether the rows are correct or if they are useful, only that they are different from the other rows in the same dataset. 

Where there are erroneous patterns, it often occurs because the tests compare two or more columns that are not really comparable, though the algorithm has no way to intuit this. This is an issue with all outlier detection tools, which treat all columns as being directly comparable. DataConsistencyChecker does have the advantage that it makes these cases plain, and allows calling clear_issues() to remove these where desired. 

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

## Sort Order for Examined Data
Where applicable, the dataframe should be sorted before running any tests. In many cases, there is no natural order to the data, but where there is, several of the tests check for patterns such as columns containing consistently increasing or decreasing values, repeating patterns in columns, and so on. Where the data is, for example, collected from SQL before running DataConsistencyChecker, the row order may other than the natural order of the data, and sorting by the relevant columns first will be useful.

## Code / ID values
Some tests on string columns are specific to columns with code or ID values, such that the individual characters in the values may have meaning. For example, with value X7333, it may be relevant that the first character is an 'X', or that the subsequent characters are 4 numeric characters. Or if two columns in a row have values X39 and X42, while another row has BB and BA, it may be relevant that in the first row, both start with X and both have two numeric digits, while the two values in the next row both have two uppercase alphabetic values, both starting with B. 

Where it is known that no columns represent ID or code values, these tests may be skipped, setting include_code_tests to False in check_data_quality(). As well, clear_results() may be called with clear_code_tests set to True to clear any results related to these tests if check_data_quality() is called including them, but subsequent evaluation finds no meaningful patterns of this type. The API get_tests_for_codes() may be called to get a list of the tests of this type.  

## Synthetic Data
The tool provides an API to generate synthetic data, which has been designed to provide columns that will be flagged by the tests, such that each test flags at least one column or set of columns. This may be useful for new users to help understand the tests, as they provide useful examples, and may be altered to determine when specifically the tests flag or do not flag patterns. The synthetic data is also used by the unit tests to help ensure consistent behaviour as the tests are expanded and improved. 

## Unit Tests
The unit tests are organized such that each test .py file tests one DataQualityChecker test, both on real and synthetic data. 

### Unit Tests on Synthetic data with null values
As null values are common in real-world data, each test must be able to handle null values in a defined manner. For most tests, the null values do not support or violate the patterns discovered. 

### Synthetic data on all columns vs those specific to the tests
When generating synthetic data, a set of columns, typically two to five columns, will be created in order to exercise the related test. Each set of columns is relevant to only one test, though, it created, will increase the test coverage of the other tests. For example, to test FEW_WITHIN_RANGE, the synthetic data creates three columns: 'few in range rand', 'few in range all', and ''few in range most'. 

It is possible to call check_data_quality() on either a specific set of tests, or without specifying a set of tests, which will result in all tests being executed. Where only a subset of the available tests are executed, it may be desired to generate only the columns relevant to these tests. This will result in faster execution times and more predictable results. It is also possilble to specify to create all synthetic columns, including those designed for other tests. These columns will be covered as well by any tests executed and will provide more testing. Typically any columns flagged by a test will be flagged regardless if other synthetic columns are generated or not. 

