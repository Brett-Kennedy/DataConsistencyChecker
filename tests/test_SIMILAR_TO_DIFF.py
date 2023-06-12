import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SIMILAR_TO_DIFF'
random.seed(0)

synth_patterns_cols = [
	'"similar_to_diff rand_a" AND "similar_to_diff rand_b" AND "similar_to_diff all"'
]
synth_exceptions_cols = [
	'"similar_to_diff rand_a" AND "similar_to_diff rand_b" AND "similar_to_diff most"'
]


def test_real():
	res = build_default_results()
	res['jm1'] = (0, ['"n" AND "total_Opnd" AND "total_Op"'])
	res['shuttle'] = ([
		'"A3" AND "A7" AND "A1"',
		'"A3" AND "A5" AND "A8"'], 0)
	res['mc1'] = ([
		'"LOC_EXECUTABLE" AND "LOC_TOTAL" AND "LOC_CODE_AND_COMMENT"',
		'"HALSTEAD_LENGTH" AND "NUM_OPERATORS" AND "NUM_OPERANDS"'
		], 0)
	res['pc1'] = (0, [
		'"loc" AND "lOCode" AND "locCodeAndComment"',
		'"N" AND "total_Opnd" AND "total_Op"'
	])
	res['cardiotocography'] = (['"V17" AND "V18" AND "V16"'], 0)
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
		0,
		0)


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
