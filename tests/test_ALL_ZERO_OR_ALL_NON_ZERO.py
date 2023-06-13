import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'ALL_ZERO_OR_ALL_NON_ZERO'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = [
	'"all_zero_or_not rand_a" AND "all_zero_or_not all" AND "all_zero_or_not most"'
]


def test_real():
	res = build_default_results()
	res['bioresponse'] = (15, [])
	res['jm1'] = ([], ['"n" AND "v" AND "l" AND "d" AND "i" AND "e" AND "t" AND "lOCode" AND "uniq_Op" AND "uniq_Opnd" AND "total_Op" AND "total_Opnd"'])
	res['nomao'] = ([
		'"V1" AND "V2"',
		'"V4" AND "V6"'
	], [])
	res['mc1'] = ([
		'"CONDITION_COUNT" AND "DECISION_COUNT" AND "MODIFIED_CONDITION_COUNT" AND "MULTIPLE_CONDITION_COUNT"'],
		[])
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
		0)           # Given the null values, there are not sufficient zeros in the columns


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
		synth_patterns_cols,
		[])


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

