# Full API

   #### &nbsp;&nbsp;&nbsp; The main APIs
   These APIs must be called for any analysis. These create a DataConsistencyChecker object and use this to evaluate a dataset. The following APIs may be used to view the results. 
- [DataConsistencyChecker](#DataConsistencyChecker)
- [init_data](#init_data)
- [check_data_quality](#check_data_quality)
- [check_data_quality_by_feature_pairs](#check_data_quality_by_feature_pairs)
   #### Methods to generate or modify test data
- [generate_synth_data](#generate_synth_data)
- [modify_real_data](#modify_real_data)
   #### Methods to output information about the tool
  These are unrelated to any specific dataset or test execution
- [get_test_list](#get_test_list)
- [get_test_descriptions](#get_test_descriptions)
- [print_test_descriptions](#print_test_descriptions)
- [get_patterns_shortlist](#get_patterns_shortlist)
- [get_tests_for_codes](#get_tests_for_codes)
- [demo_test](#demo_test)
    #### Methods to output statistics about the dataset, unrelated to any tests executed
- [display_columns_types_list](#display_columns_types_list)
- [display_columns_types_table](#display_columns_types_table)
    #### Methods to output summaries of the results
  These provide counts of the numbers of rows and features identified as patterns with and without exceptions.
- [get_test_ids_with_results](#get_test_ids_with_results)
- [get_single_feature_tests_matrix](#get_single_feature_tests_matrix)
- [get_patterns_list](#get_patterns_list)
- [get_exceptions_list](#get_exceptions_list)
- [get_exceptions_by_column](#get_exceptions_by_column)
- [summarize_patterns_by_test_and_feature](#summarize_patterns_by_test_and_feature)
- [summarize_exceptions_by_test_and_feature](#summarize_exceptions_by_test_and_feature)
- [summarize_patterns_by_test](#summarize_patterns_by_test)
- [summarize_exceptions_by_test](#summarize_exceptions_by_test)
- [summarize_patterns_and_exceptions](#summarize_patterns_and_exceptions)
- [plot_final_scores_distribution_by_row](#plot_final_scores_distribution_by_row)
- [plot_final_scores_distribution_by_feature](#plot_final_scores_distribution_by_feature)
- [plot_final_scores_distribution_by_test](#plot_final_scores_distribution_by_test)
- [quick_report](#quick_report)
  #### Methods to output the results
  These provide information about the specific patterns and exceptions found.
- [display_detailed_results](#display_detailed_results)
- [display_next](#display_next)
  #### Methods to describe specific rows in terms of their patterns and exceptions
  These summarize specific rows and compare rows to each other
- [get_outlier_scores](#get_outlier_scores)
- [get_results_by_row_id](#get_results_by_row_id)
- [display_least_flagged_rows](#display_least_flagged_rows)
- [display_most_flagged_rows](#display_most_flagged_rows)
  #### Methods to find relationships between the data and the numbers of issues found
- [plot_columns_vs_final_scores](#plot_columns_vs_final_scores)
    

## DataConsistencyChecker
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

## init_data
**init_data**(df, known_date_cols=None)

        df: dataframe to be assessed

        known_date_cols: list of strings
            If specified, these, and only these, columns will be treated as date columns.

## check_data_quality
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

## check_data_quality_by_feature_pairs
**check_data_quality_by_feature_pairs**(max_features_shown=30)

      An alternative to check_data_quality(). This runs similar (though fewer) tests on pairs of features and, for
      each test, presents a matrix indicating for what fraction of the rows a given relationship between the features
      holds true.

        max_features_shown: int
            Where there are many features, it can be infeasible to show a heatmap of all features. However, it may be
            useful to render a heatmap for the first features. max_features_shown specifies how many features, at most,
            will be included in the heatmaps rendered.


## Methods to generate or modify test data

## generate_synth_data
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

## modify_real_data
**modify_real_data**(df, num_modifications=5)

        Given a real dataset, modify it slightly, in order to add what are likely inconsistencies to the data

        df: pandas dataframe
            A real or synthetic dataset.

        num_modifications: int
            The number of modifications to make. This should be small, so as not to change the overall distribution
            of the data

        Returns:
            the modified dataframe,
            a list of row numbers and column names, indicating the cells that were modified


## Methods to output information about the tool

## get_test_list
**get_test_list**():
        
        Returns a python list, listing each test available by ID.

## get_test_descriptions
**get_test_descriptions**()
        
        Returns a python dictionary, with each test ID as key, matched with a short text explanation of the test.
        
## print_test_descriptions
**print_test_descriptions**(long_desc=False)
        
        Prints to screen a prettified list of tests and their descriptions.

        long_desc: bool
            If True, longer descriptions will be displayed for tests where available. If False, the short descriptions
            only will be displayed.
        
## get_patterns_shortlist
**get_patterns_shortlist**()
        
        Returns an array with the IDs of the tests in the short list. These are the tests that will be presented by
        default when calling get_patterns() to list the patterns discovered.
        
## get_tests_for_codes
**get_tests_for_codes**()
        
        Returns an array with the IDs of the tests that related to ID and code values. Where it is known that no columns
        are of this type, these tests may be skipped.
        

## demo_test
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
        

## Methods to output statistics about the dataset, unrelated to any tests executed

## display_columns_types_list
**display_columns_types_list**()
        
        Displays, for each of the four column types identified by the tool, which columns in the data are of those
        types. This may be called to check the column types were identified correctly. This will skip columns removed
        from analysis due to having only one unique value.
        
## display_columns_types_table
**display_columns_types_table**()
        
        Displays the first rows of the data, along with the identified column type in the first row. Similar to
        calling display_columns_types_list(), this may be used to determine if the inferred column types are correct.
        
        
## Methods to output summaries of the results

## get_test_ids_with_results
**get_test_ids_with_results**(include_patterns=True, include_exceptions=True)
        
        Gets a list of test ids, which may be used, for example, to loop through tests calling other APIs such as
        display_detailed_results(). These are the ids of tests that flagged at least one pattern and/or exeception.

        include_patterns: bool
            If True, the returned list will include all test ids for tests that flagged at least one pattern without
            exceptions

        include_exceptions: bool
            If True, the returned list will include all test ids for tests that flagged at least one pattern with
            exceptions

        Returns: array of test ids
            Returns an array of test ids, where each test found at least one pattern, with or without exceptions, as
            specified
        
## get_single_feature_tests_matrix      
**get_single_feature_tests_matrix**():
        
        Returns a matrix with a column for every column in the original data and a row for each test executed. The cell
        values contain the percent of rows matching the pattern associated with the test. In many cases, this will be
        NaN (rendered as '-'), as not all tests execute on all columns. For example, tests for large numeric values
        execute only on numeric columns. As well, for efficiency, many tests skip examining columns that have many null
        values, many zero values, few or many unique values, etc. This in necessary to make the execution on many tests
        tractable where there are many columns and/or many rows.
        
## get_patterns_list
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

## get_exceptions_list
**get_exceptions_list**()
        
        Returns a dataframe containing a row for each pattern that was discovered with exceptions. This has a similar
        format to the dataframe returned by get_patterns_list(), with one additional column representing the number
        of exceptions found. The dataframe has columns for: test id, the set of columns involved in the pattern,
        a description of the pattern and exceptions, and the number of exceptions.
        
## get_exceptions
**get_exceptions**()
        
        Returns a dataframe with the same set of rows as the original dataframe, but a column for each pattern that was
        discovered that had exceptions, and a column indicating the final score for each row. This dataframe can be very 
        large and is not generally useful to display, but may be collected for further analysis.

        Returns:
            pandas.DataFrame: DataFrame containing exceptions and final scores
        
## get_exceptions_by_column
**get_exceptions_by_column**()
        
        Returns a dataframe with the same shape as the original dataframe, but with each cell containing, instead
        of the original value for each feature for each row, a score allocated to that cell. Each pattern with
        exceptions has a score of 1.0, but patterns the cover multiple columns will give each cell a fraction of this.
        For example with a pattern covering 4 columns, any cells that are flagged will receive a score of 0.25 for
        this pattern. Each cell will have the sum of all patterns with exceptions where they are flagged.
        
## summarize_patterns_by_test_and_feature      
**summarize_patterns_by_test_and_feature**(all_tests=False, heatmap=False)
        
        Create and return a dataframe with a row for each test and a column for each feature in the original data. Each
        cell has a 0 or 1, indicating if the pattern was found without exceptions in that feature. Note, some tests to
        not identify patterns, such as VERY_LARGE. The dataframe is returned, and optionally displayed as a heatmap.

        all_tests: bool
            If all_tests is True, all tests are included in the output, even those that found no patterns. This may be
            used specifically to check which found no patterns.

        heatmap: bool
            If True, a heatmap will be displayed
        
## summarize_exceptions_by_test_and_feature
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
        
## summarize_patterns_by_test
**summarize_patterns_by_test**(heatmap=False)
        
        Create and return a dataframe with a row for each test, indicating the number of features where the pattern
        was found.

        heatmap: bool
            If True, a heatmap of the results are displayed

## summarize_exceptions_by_test
**summarize_exceptions_by_test**(heatmap=False)
        
        Create and return a dataframe with a row for each test, indicating 1) the number of features where the pattern
        was found but also exceptions, 2) the number of issues total flagged (across all features and all rows).

        heatmap: bool
            If True, a heatmap of the results are displayed

## summarize_patterns_and_exceptions               
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

## plot_final_scores_distribution_by_row
**plot_final_scores_distribution_by_row**()
        
        Display a probability plot and histogram representing the distribution of final scores by row.
        
## plot_final_scores_distribution_by_feature
**plot_final_scores_distribution_by_feature**()
        
        Display a bar plot representing the distribution of final scores by feature.
        
## plot_final_scores_distribution_by_test
 **plot_final_scores_distribution_by_test**()
        
        Display a bar plot representing the distribution of final scores by test.

## quick_report
**quick_report**()
        
        A convenience method, which calls several other APIs, to give an overview of the results in a single API.

## Methods to output the results
               
## display_detailed_results
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
            max_shown=-1,
            save_to_disk=False,
            output_folder=None
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

        save_to_disk: bool
            If set True, output will be written to a file on disk

        output_folder: str
            Used only if save_to_disk is True. Indicates the file path to use for the output. If not specified, the
            current folder will be used.
        
## display_next
**display_next**()
        
        This may be used where there are many results, and we wish to view detailed descriptions of all or most of
        these. This API calls display_detailed_results() for one test at a time, for each test that identified at least
        one pattern (with or without exceptions) during the last call to check_data_quality(). This allows, when working
        with notebooks, for output to be spread over multiple cells, which can make viewing it simpler. Note though,
        where many tests flag patterns, in most cases only a subset of these would be useful to examine in detail,
        though this varies for different projects.

## Methods to describe specific rows in terms of their patterns and exceptions        

## get_outlier_scores
**get_outlier_scores**()
        
        Returns an outlier score for each row, similar to most outlier detectors.
        Returns a python array with an element for each row in the original data. All values are non-negative integer
        values, with most rows containing zero for most datasets. The scores indicate the number of tests that flagged
        each row. 

## get_results_by_row_id
**get_results_by_row_id**(row_num)
        
        Returns a list of tuples, with each tuple containing a test ID, and column name, for all issues flagged in the
        specified row.
        
## display_least_flagged_rows
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
        
## display_most_flagged_rows
**display_most_flagged_rows**(with_results=True, n_rows=10)
        
        This is similar to display_least_flagged_rows, but displays the rows with the most identified issues.

        with_results: bool
            If True, the flagged rows will be display in separate tables, with additional rows indicating which tests
            flagged which columns. If False, all displayed rows will be displayed in a single table; the flagged
            columns will be highlighted, but there will not be an indication of which tests flagged them.
            
        n_rows: int
            The maximum number of original rows to present.

## Methods to find relationships between the data and the numbers of issues found

## plot_columns_vs_final_scores
**plot_columns_vs_final_scores**()
        
        Used to determine if there are any relationships between column values and the final scores of the rows. This
        displays tables and plots presenting any relationships found.
        
