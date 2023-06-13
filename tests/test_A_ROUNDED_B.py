import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'A_ROUNDED_B'
random.seed(0)


# Currently none pass
def test_real():
	res = build_default_results()
	real_test(test_id, res)


synth_patterns_cols = [
	'"a_rounded_b rand" AND "a_rounded_b all_a"',
	'"a_rounded_b rand" AND "a_rounded_b all_b"',
	'"a_rounded_b rand" AND "a_rounded_b all_c"',
	'"a_rounded_b rand" AND "a_rounded_b all_d"',
	'"a_rounded_b rand" AND "a_rounded_b all_e"',
	'"a_rounded_b rand" AND "a_rounded_b all_f"',
	'"a_rounded_b all_a" AND "a_rounded_b all_f"',
	'"a_rounded_b all_b" AND "a_rounded_b all_f"',
	'"a_rounded_b all_c" AND "a_rounded_b all_f"'
	]

synth_exceptions_cols = [
	'"a_rounded_b rand" AND "a_rounded_b most_a"',
	'"a_rounded_b rand" AND "a_rounded_b most_b"',
	'"a_rounded_b rand" AND "a_rounded_b most_c"',
	'"a_rounded_b rand" AND "a_rounded_b most_d"',
	'"a_rounded_b all_a" AND "a_rounded_b all_e"',
	'"a_rounded_b all_b" AND "a_rounded_b all_e"',
	'"a_rounded_b most_b" AND "a_rounded_b all_e"',
	'"a_rounded_b all_c" AND "a_rounded_b all_e"',
	'"a_rounded_b rand" AND "a_rounded_b most_e"',
	'"a_rounded_b all_b" AND "a_rounded_b most_e"',
	'"a_rounded_b most_b" AND "a_rounded_b most_e"',
	'"a_rounded_b most_a" AND "a_rounded_b all_f"',
	'"a_rounded_b most_b" AND "a_rounded_b all_f"',
	'"a_rounded_b most_c" AND "a_rounded_b all_f"',
	'"a_rounded_b all_d" AND "a_rounded_b all_f"',
	'"a_rounded_b most_d" AND "a_rounded_b all_f"',
	'"a_rounded_b rand" AND "a_rounded_b most_f"',
	'"a_rounded_b all_a" AND "a_rounded_b most_f"',
	'"a_rounded_b most_a" AND "a_rounded_b most_f"',
	'"a_rounded_b all_b" AND "a_rounded_b most_f"',
	'"a_rounded_b most_b" AND "a_rounded_b most_f"',
	'"a_rounded_b all_c" AND "a_rounded_b most_f"',
	'"a_rounded_b most_c" AND "a_rounded_b most_f"',
	'"a_rounded_b all_d" AND "a_rounded_b most_f"',
	'"a_rounded_b most_d" AND "a_rounded_b most_f"'
]


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
		synth_exceptions_cols,
		allow_more=True,
	)


def test_synthetic_in_sync_nulls():  # currently failing. I think we should have the same patterns if the nones are in sync
	synth_test(
		test_id,
		'in-sync',
		synth_patterns_cols,
		synth_exceptions_cols,
		allow_more=True
	)


def test_synthetic_random_nulls():
	synth_test(
		test_id,
		'random',
		-5,
		-5,
		allow_more=True
	)


def test_synthetic_80_percent_nulls():
	synth_test(
		test_id,
		'80-percent',
		-5,
		-5
	)


def test_synthetic_all_cols_no_nulls():
	synth_test_all_cols(
		test_id,
		'none',
		synth_patterns_cols,
		synth_exceptions_cols
	)


def test_synthetic_all_cols_one_row_nulls():
	synth_test_all_cols(
		test_id,
		'one-row',
		synth_patterns_cols,
		synth_exceptions_cols
	)


def test_synthetic_all_cols_in_sync_nulls():
	synth_test_all_cols(
		test_id,
		'in-sync',
		synth_patterns_cols,
		synth_exceptions_cols
	)


def test_synthetic_all_cols_random_nulls():
	synth_test_all_cols(
		test_id,
		'random',
		synth_patterns_cols,
		synth_exceptions_cols
	)


def test_synthetic_all_cols_80_percent_nulls():
	synth_test_all_cols(
		test_id,
		'80-percent',
		synth_patterns_cols,
		synth_exceptions_cols
	)

