# Unit Tests

The unit tests are organized such that each test .py file tests one DataQualityChecker test, both on real and synthetic data. 

### Unit Tests on Synthetic data with null values
As null values are common in real-world data, each test must be able to handle null values in a defined manner. For most tests, the null values do not support or violate the patterns discovered. 

### Synthetic data on all columns vs those specific to the tests
When generating synthetic data, a set of columns, typically two to five columns, will be created in order to exercise the related test. Each set of columns is relevant to only one test, though, it created, will increase the test coverage of the other tests. For example, to test FEW_WITHIN_RANGE, the synthetic data creates three columns: 'few in range rand', 'few in range all', and ''few in range most'. 

It is possible to call check_data_quality() on either a specific set of tests, or without specifying a set of tests, which will result in all tests being executed. Where only a subset of the available tests are executed, it may be desired to generate only the columns relevant to these tests. This will result in faster execution times and more predictable results. It is also possilble to specify to create all synthetic columns, including those designed for other tests. These columns will be covered as well by any tests executed and will provide more testing. Typically any columns flagged by a test will be flagged regardless if other synthetic columns are generated or not. 

