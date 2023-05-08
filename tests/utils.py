import pandas as pd
import random
import sklearn.datasets as datasets
from sklearn.datasets import fetch_openml

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker
from _list_real_files import real_files


def build_default_results():
	d = {dataset: ([], []) for dataset in real_files}
	return d


def real_test(test_id, expected_results_dict):
	for dataset_name in real_files:
		expected_patterns_cols = expected_results_dict[dataset_name][0]
		expected_exceptions_cols = expected_results_dict[dataset_name][1]

		print(f"Testing {dataset_name}")
		fast_only = dataset_name in ['Satellite']
		try:
			data = fetch_openml(dataset_name, version=1)
		except:
			# Most datasets have a version 1, but if not, just load any version. This may result in a warning.
			data = fetch_openml(dataset_name)
		data_df = pd.DataFrame(data.data, columns=data.feature_names)

		dc = DataConsistencyChecker(verbose=-1, execute_list=[test_id])
		dc.check_data_quality(data_df, fast_only=fast_only)
		patterns_df = dc.get_patterns_summary(short_list=True)
		assert ((patterns_df is None) and (len(expected_patterns_cols) == 0)) or \
				(len(patterns_df) == len(expected_patterns_cols))
		for col in expected_patterns_cols:
			assert col in patterns_df['Column(s)'].values

		exceptions_df = dc.get_exceptions_summary()
		assert ((exceptions_df is None) and (len(expected_patterns_cols) == 0)) or \
				(len(exceptions_df) == len(expected_exceptions_cols))
		for col in expected_exceptions_cols:
			assert col in exceptions_df['Column(s)'].values


def synth_test(test_id, add_nones, expected_patterns_cols, expected_exceptions_cols, allow_more=False):
	dc = DataConsistencyChecker(verbose=-1, execute_list=[test_id])
	synth_df = dc.generate_synth_data(all_cols=False, add_nones=add_nones)
	dc.check_data_quality(synth_df)

	ret = dc.get_patterns_summary(short_list=False)
	if allow_more:
		assert len(ret) >= len(expected_patterns_cols)
	else:
		assert len(ret) == len(expected_patterns_cols)
	for col in expected_patterns_cols:
		assert col in ret['Column(s)'].values

	ret = dc.get_exceptions_summary()
	if allow_more:
		assert len(ret) >= len(expected_exceptions_cols)
	else:
		assert len(ret) == len(expected_exceptions_cols)
	for col in expected_exceptions_cols:
		assert col in ret['Column(s)'].values


def synth_test_all_cols(test_id, add_nones, patterns_cols, exceptions_cols):
	# dc = DataConsistencyChecker(execute_list=[test_id])
	# synth_df = dc.generate_synth_data(all_cols=True, add_nones=add_nones)
	# dc.check_data_quality(synth_df)
	#
	# ret = dc.get_patterns_summary(short_list=False)
	# for col in patterns_cols:
	# 	assert col in ret['Column(s)'].values
	#
	# ret = dc.get_exceptions_summary()
	# for col in exceptions_cols:
	# 	assert col in ret['Column(s)'].values
	pass
