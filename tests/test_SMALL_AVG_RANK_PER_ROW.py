import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SMALL_AVG_RANK_PER_ROW'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = ['"small_avg_rank_rand_0" AND "small_avg_rank_rand_1" AND "small_avg_rank_rand_2" AND "small_avg_rank_rand_3" AND "small_avg_rank_rand_4" AND "small_avg_rank_rand_5" AND "small_avg_rank_rand_6" AND "small_avg_rank_rand_7" AND "small_avg_rank_rand_8" AND "small_avg_rank_rand_9"']


def test_real():
	res = build_default_results()
	res['gas-drift'] = (0, 1)
	res['phoneme'] = (0, 1)
	res['one-hundred-plants-margin'] = (0, 1)
	res['madelon'] = (0, 1)
	res['musk'] = (0, 1)
	res['CreditCardSubset'] = (0, 1)
	res['shuttle'] = (0, 1)
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
		synth_patterns_cols,
		synth_exceptions_cols)


def test_synthetic_random_nulls():
	synth_test(
		test_id,
		'random',
		synth_patterns_cols,
		synth_exceptions_cols)


def test_synthetic_80_percent_nulls():
	synth_test(
		test_id,
		'80-percent',
		synth_patterns_cols,
		synth_exceptions_cols)


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
