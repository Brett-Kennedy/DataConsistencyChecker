import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'LARGE_GIVEN_PAIR'
random.seed(0)

synth_patterns_cols = 0  # None
synth_exceptions_cols = [
	'"large_given_pair rand_a" AND "large_given_pair rand_b" AND "large_given_pair most"',
	'"large_given_pair rand_a" AND "large_given_pair rand_b" AND "large_given_pair date_most"'
]


def test_real():
	res = build_default_results()
	res['SpeedDating'] = ([], 11583)
	res['eucalyptus'] = (0, ['"Locality" AND "Latitude" AND "DBH"'])
	res['credit-approval'] = (0, ['"A4" AND "A5" AND "A3"', '"A4" AND "A5" AND "A8"'])
	res['adult'] = (0, 47)
	res['anneal'] = (0, 5)
	res['credit-g'] = ([], 38)
	res['bank-marketing'] = ([], 25)
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
		synth_patterns_cols,
		synth_exceptions_cols)


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
		synth_patterns_cols,
		synth_exceptions_cols)


def test_synthetic_all_cols_80_percent_nulls():
	synth_test_all_cols(
		test_id,
		'80-percent',
		synth_patterns_cols,
		synth_exceptions_cols)
