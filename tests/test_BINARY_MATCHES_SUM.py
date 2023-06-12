import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'BINARY_MATCHES_SUM'
random.seed(0)

synth_patterns_cols = [
	'"bin_match_sum rand_a" AND "bin_match_sum rand_b" AND "bin_match_sum all"'
]
synth_exceptions_cols = [
	'"bin_match_sum rand_a" AND "bin_match_sum rand_b" AND "bin_match_sum most"'
]


def test_real():
	res = build_default_results()
	res['nomao'] = ([
		'"V1" AND "V81" AND "V7"',
		'"V1" AND "V82" AND "V7"',
		'"V1" AND "V83" AND "V7"',
		'"V1" AND "V84" AND "V7"',
		'"V1" AND "V85" AND "V7"',
		'"V1" AND "V86" AND "V7"',
		'"V1" AND "V2" AND "V8"',
		'"V1" AND "V3" AND "V8"',
		'"V1" AND "V4" AND "V8"',
		'"V2" AND "V3" AND "V8"',
		'"V2" AND "V4" AND "V8"',
		'"V2" AND "V6" AND "V8"',
		'"V2" AND "V81" AND "V8"',
		'"V2" AND "V82" AND "V8"',
		'"V2" AND "V83" AND "V8"',
		'"V2" AND "V84" AND "V8"',
		'"V2" AND "V85" AND "V8"',
		'"V2" AND "V86" AND "V8"',
		'"V3" AND "V4" AND "V8"',
		'"V3" AND "V6" AND "V8"',
		'"V3" AND "V81" AND "V8"',
		'"V3" AND "V82" AND "V8"',
		'"V3" AND "V83" AND "V8"',
		'"V3" AND "V84" AND "V8"',
		'"V3" AND "V85" AND "V8"',
		'"V3" AND "V86" AND "V8"',
		'"V4" AND "V6" AND "V8"',
		'"V4" AND "V81" AND "V8"',
		'"V4" AND "V82" AND "V8"',
		'"V4" AND "V83" AND "V8"',
		'"V4" AND "V84" AND "V8"',
		'"V4" AND "V85" AND "V8"',
		'"V4" AND "V86" AND "V8"',
	],
	[
		'"V1" AND "V93" AND "V7"',
		'"V1" AND "V94" AND "V7"',
		'"V1" AND "V95" AND "V7"',
		'"V1" AND "V109" AND "V7"',
		'"V1" AND "V113" AND "V7"',
		'"V1" AND "V118" AND "V7"',
		'"V1" AND "V94" AND "V8"',
		'"V2" AND "V33" AND "V8"',
		'"V2" AND "V93" AND "V8"',
		'"V2" AND "V94" AND "V8"',
		'"V2" AND "V95" AND "V8"',
		'"V2" AND "V101" AND "V8"',
		'"V2" AND "V105" AND "V8"',
		'"V2" AND "V109" AND "V8"',
		'"V2" AND "V113" AND "V8"',
		'"V2" AND "V117" AND "V8"',
		'"V2" AND "V118" AND "V8"',
		'"V3" AND "V94" AND "V8"',
		'"V3" AND "V95" AND "V8"',
		'"V3" AND "V105" AND "V8"',
		'"V3" AND "V109" AND "V8"',
		'"V3" AND "V113" AND "V8"',
		'"V3" AND "V117" AND "V8"',
		'"V3" AND "V118" AND "V8"',
		'"V4" AND "V93" AND "V8"',
		'"V4" AND "V94" AND "V8"',
		'"V4" AND "V95" AND "V8"',
		'"V4" AND "V101" AND "V8"',
		'"V4" AND "V105" AND "V8"',
		'"V4" AND "V109" AND "V8"',
		'"V4" AND "V113" AND "V8"',
		'"V4" AND "V117" AND "V8"',
		'"V4" AND "V118" AND "V8"',
		'"V6" AND "V93" AND "V8"',
		'"V6" AND "V94" AND "V8"',
		'"V6" AND "V95" AND "V8"',
	])
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
		2)  # The pattern is now an exception


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
