import pandas as pd
import random
import sklearn.datasets as datasets
from sklearn.datasets import fetch_openml

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker


def synth_test(test_id, add_nones, patterns_cols, exceptions_cols):
	dc = DataConsistencyChecker(execute_list=[test_id])
	synth_df = dc.generate_synth_data(all_cols=False, add_nones=add_nones)
	dc.check_data_quality(synth_df)

	ret = dc.get_patterns_summary(short_list=False)
	assert len(ret) == len(patterns_cols)
	for col in patterns_cols:
		assert col in ret['Column(s)'].values

	ret = dc.get_exceptions_summary()
	assert len(ret) == len(exceptions_cols)
	for col in exceptions_cols:
		assert col in ret['Column(s)'].values


def synth_test_all_cols(test_id, add_nones, patterns_cols, exceptions_cols):
	dc = DataConsistencyChecker(execute_list=[test_id])
	synth_df = dc.generate_synth_data(all_cols=True, add_nones=add_nones)
	dc.check_data_quality(synth_df)

	ret = dc.get_patterns_summary(short_list=False)
	for col in patterns_cols:
		assert col in ret['Column(s)'].values

	ret = dc.get_exceptions_summary()
	for col in exceptions_cols:
		assert col in ret['Column(s)'].values

def real_test(df, test_id, patterns_cols, exceptions_cols):
	dc = DataConsistencyChecker(execute_list=[test_id])
	dc.check_data_quality(df)

	ret = dc.get_patterns_summary(short_list=False)
	assert len(ret) == len(patterns_cols)
	for col in patterns_cols:
		assert col in ret['Column(s)'].values

	ret = dc.get_exceptions_summary()
	assert len(ret) == len(exceptions_cols)
	for col in exceptions_cols:
		assert col in ret['Column(s)'].values


def kropt_test(test_id, patterns_cols, exceptions_cols):
	data = fetch_openml("kropt")
	df = pd.DataFrame(data.data, columns=data.feature_names)
	real_test(df, test_id, patterns_cols, exceptions_cols)