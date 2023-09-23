import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SAME_VALUES'
random.seed(0)

synth_patterns_cols = ['"same rand" AND "same all"']
synth_exceptions_cols = [
	'"same rand" AND "same most"',
	'"same all" AND "same most"',
	'"same rand_date" AND "same most_date"'
]


def test_real():
	res = build_default_results()
	res['cnae-9'] = (['"V383" AND "V602"'], 0)
	res['nomao'] = (18, 4)
	res['hypothyroid'] = ([], ['"T4U_measured" AND "FTI_measured"'])
	res['cardiotocography'] = (['"V4" AND "V5"'], [])
	res['allbp'] = ([], ['"T4U_measured" AND "FTI_measured"'])
	res['allrep'] = ([], ['"T4U_measured" AND "FTI_measured"'])
	res['dis'] = ([], ['"T4U_measured" AND "FTI_measured"'])
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
