import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'DECISION_TREE_REGRESSOR'
random.seed(0)

synth_patterns_cols = 1
	#'"dt regr. 1b" AND "dt regr. 1a" AND "dt regr. 2"'
synth_exceptions_cols = 1
	#'"dt regr. 1d" AND "dt regr. 1c" AND "dt regr. 3"'


def test_real():
	res = build_default_results()
	res['bioresponse'] = (0, 1)  # '"D19" AND "D49" AND "D183"'
	real_test(test_id, res)


def test_synthetic_no_nulls():
	synth_test(
		test_id,
		'none',
		synth_patterns_cols,
		synth_exceptions_cols)


def test_synthetic_one_row_nulls():
	synth_test(
		test_id,
		'one-row',
		0,
		2)  # The pattern is now an exception


def test_synthetic_in_sync_nulls():
	synth_test(
		test_id,
		'in-sync',
		0,
		0)


def test_synthetic_random_nulls():
	synth_test(
		test_id,
		'random',
		0,
		0)


def test_synthetic_80_percent_nulls():
	synth_test(
		test_id,
		'80-percent',
		0,
		0)


def test_synthetic_all_cols_no_nulls():
	synth_test_all_cols(
		test_id,
		'none',
		synth_patterns_cols,
		synth_exceptions_cols)


def test_synthetic_all_cols_one_row_nulls():
	synth_test_all_cols(
		test_id,
		'one-row',
		synth_patterns_cols,
		synth_exceptions_cols)


def test_synthetic_all_cols_in_sync_nulls():
	synth_test_all_cols(
		test_id,
		'in-sync',
		synth_patterns_cols,
		synth_exceptions_cols)


def test_synthetic_all_cols_random_nulls():
	synth_test_all_cols(
		test_id,
		'random',
		0,
		0)


def test_synthetic_all_cols_80_percent_nulls():
	synth_test_all_cols(
		test_id,
		'80-percent',
		0,
		0)
