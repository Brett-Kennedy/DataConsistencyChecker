Overview
===================================

Introduction
----------------------------

A python tool to examine datasets for consistency. It performs approximately 150 tests, identifying patterns in the data and any exceptions to these. The tool provides useful analysis for any EDA (Exploratory Data Analysis) and may be useful for interpretable outlier detection.

Background
-------------

The central idea of the tool is to automate the tests that would be done by a person examining a dataset and trying to determine the patterns within the columns, between the columns, and between the rows (in cases where there is some meaning to the order of the rows). That is, the tool essentially does what a person, studying a new dataset, would do, but automatically. This does not remove the need to do manual data exploration and, where the goal is identifying outliers, other forms of outlier detection, but does cover much of the work identifying the patterns in the data and the exceptions to these, with the idea that these are two of the principle tasks to understand a dataset. The tool allows users to perform this work quicker, more exhaustively, and covering more tests than would normally be done.

The tool executes a large set of tests over a dataset. For example, one test examines the number of decimal digits typically found in a numeric column. If the data consistently contains, say, four decimal digits, with no exceptions, this will be reported as a strong pattern without exceptions; if the data in this column nearly always has four decimal digits, with a small number of expections (for example having eight decimal digits), this will be reported as a strong pattern with exceptions. In this example, this suggests the identified rows may have been collected or processed in a different manner than the other rows. And, while this may not be interesting in itself, where rows are flagged multiple times for this or other issues, users can be more confident the rows are somehow different.

The tests run over single columns (for example, checking for rare, unsually small or large values, etc), pairs of columns (for example checking if one column is consistently larger than the other, contains similar characters (in the case of code or ID string values), etc.), or larger sets of columns (for example, checking if one column tends to be the sum, or mean of another set of columns).

The unusual data found may be due to data collection errors, to mixing different types of data together, or other issues that may be considered errors, or that may be informative.

While many of the individual tests are straight-forward, there are real advantages to running them within a single package, notably the ability to identify rows that are significantly different from the majority, even where this is only evident from multipe subtle deviations, and the ability to ammortize processing work over many tests. Much of the computation necessary for the tests is shared among two or more tests and consequently running many tests on the same dataset can result in increased performance, in terms of time per test.

EDA
----
DataConsistencyChecker may be used for exploratory data analsys simply by running the tool and examining the patterns, and exceptions to the patterns that are found.

Exploratory data analysis may be run for a variety of purposes, but generally most relate to one of two high-level purposes: ensuring the data is of sufficient quality to build a model (or can be made to be of sufficient quality after removing or replacing any necessary values, rows, or columns), and gaining insights into the data in order to build appropriate models. This tool may be of assistence for both these purposes.

Outlier Detection
-------------------
DataConsistencyChecker provides a large set of tests, each independent of the others and each highly interpretable. Running the tool, it's common for rows that are unusual to be flagged multiple times. This allows us to evaluate the outlier-ness of any given row based on the number of times it was flagged for issues relative to the other rows in the dataset. For example, if a row contains, for example, some values that are unusually large, as well as some strings with unusual characters, strings of unusual lengths, time values at unusual times of day, and rare values, the row will then be flagged multiple times, giving it a relatively high outlier score.

The majority of outlier detectors work on either strictly numeric data, or strictly categorical data, with numeric outlier detectors seeking to identify unusual rows as points in high-dimensional space far from the majority of other rows, and with categorical outlier detectors seeking to identify unusual combinations of values. These tests are very useful, but can be uninterpretable with a sufficient number of features: it can be difficult to confirm the rows are more unusual than the majority of rows or to even determine why they were flagged as outliers.

DataConsistencyChecker may flag a row, for example, for having a very large value in Column A, a value with an unsual number of digits in Column D, an unusually long string in Column C, and values with unusual rounding in Column D and Column G. As with any outlier detection scheme, it is difficult to gauge the outlier-ness of each of these, but DataConsistencyChecker does have the desirable property that each of these is simple to comprehend.

As this tool is limited to interpretable outlier detection methods, it does not test for the multi-dimensional outliers detected by other algorithms such as Isolation Forest. These outlier detection algorithms should generally also be executed to have a full understanding of the outliers present in the data.
