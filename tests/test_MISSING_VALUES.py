import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'MISSING_VALUES'
random.seed(0)

synth_patterns_cols = ['missing vals all']
synth_exceptions_cols = ['missing vals most', 'missing vals most null']


def test_real():
	res = build_default_results()
	res['isolet'] = (617, [])
	res['hypothyroid'] = ([], ['age'])
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
		[],
		['missing vals all', 'missing vals most', 'missing vals most null'])


def test_synthetic_in_sync_nulls():
	synth_test(
		test_id,
		'in-sync',
		[],
		['missing vals most null'])  # Mostly NUll, with some non-null


def test_synthetic_random_nulls():
	synth_test(
		test_id,
		'random',
		[],
		['missing vals most null'])


def test_synthetic_80_percent_nulls():
	synth_test(
		test_id,
		'80-percent',
		[],
		['missing vals most null'])


def test_synthetic_all_cols_no_nulls():
	synth_test_all_cols(
		test_id,
		'none',
		['missing vals all'],
		['missing vals most', 'missing vals most null'])


def test_synthetic_all_cols_one_row_nulls():
	synth_test_all_cols(
		test_id,
		'one-row',
		[],
		['missing vals all', 'missing vals most', 'missing vals most null'])


def test_synthetic_all_cols_in_sync_nulls():
	synth_test_all_cols(
		test_id,
		'in-sync',
		[],
		['missing vals most null'])  # Mostly NUll, with some non-null


def test_synthetic_all_cols_random_nulls():
	synth_test_all_cols(
		test_id,
		'random',
		[],
		['missing vals most null'])


def test_synthetic_all_cols_80_percent_nulls():
	synth_test_all_cols(
		test_id,
		'80-percent',
		[],
		['missing vals most null'])

