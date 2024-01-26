# Performance

The tool typically runs in under a minute for small files and under 30 minutes even for very large files, with hundreds of thousands of rows and hundreds of columns. However, it is often useful to specify to execute the tool quicker, especially if calling it frequently, or with many datasets. Several techniques to speed the execution are listed here:

- Exclude some tests. Some may be of less interest to your project or may be slower to execute.  
- Set the max_combinations parameter to a lower value, which will skip a significant amount of processing in some cases. The default is 100,000. The value relates to the number of sets of columns examined. For example, if the test checks each triple of columns, such as with SIMILAR_TO_RATIO, then the number of sets of three numeric columns is the number of combinations. This is equal to (N choose 3), where N is the number of numeric columns. With verbose set to 2 or higher, it's possible to see the number of combinations required and the current setting of max_combinations. 
- Run on a sample of the rows. Doing this, you will not be able to properly identify exceptions to the patterns, but will be able to identify the patterns themselves. 
- Columns may be removed from the dataframe before running the tool to remove checks on these columns. 
- The fast_only parameters in check_data_quality() may be set to True. Setting this, only tests that are consistently fast will be executed.
- Set include_code_tests to False. Where it is known that none of the columns represent ID or Code values, the tests associated with these types of columns may be skipped. These tests tend to be fast to execute in any case, but removing these can save some time as well as noise from the results. 
- Set the verbose level to 2 or 3. This allows users to monitor the progress better and determine if it is acceptable to wait, or if any of the other steps listed here should be tried.
