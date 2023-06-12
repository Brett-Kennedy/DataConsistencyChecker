import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SUM_OF_COLUMNS'
random.seed(0)

synth_patterns_cols = [
	'"sum of cols rand_a" AND "sum of cols rand_b" AND "sum of cols rand_c" AND "sum of cols all"',
	'"sum of cols rand_a" AND "sum of cols rand_b" AND "sum of cols rand_c" AND "sum of cols plus all"',
	'"sum of cols rand_a" AND "sum of cols rand_b" AND "sum of cols rand_c" AND "sum of cols times all"'
]
synth_exceptions_cols = [
	'"sum of cols rand_a" AND "sum of cols rand_b" AND "sum of cols rand_c" AND "sum of cols most"',
	'"sum of cols rand_a" AND "sum of cols rand_b" AND "sum of cols rand_c" AND "sum of cols plus most"',
	'"sum of cols rand_a" AND "sum of cols rand_b" AND "sum of cols rand_c" AND "sum of cols times most"'
]


def test_real():
	res = build_default_results()
	res['mc1'] = (['"NUM_OPERANDS" AND "NUM_OPERATORS" AND "HALSTEAD_LENGTH"'], 0)
	res['pc1'] = (0, ['"total_Op" AND "total_Opnd" AND "N"'])
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
