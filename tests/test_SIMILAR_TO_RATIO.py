import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SIMILAR_TO_RATIO'
random.seed(0)

synth_patterns_cols = ['"similar_to_ratio rand_a" AND "similar_to_ratio rand_b" AND "similar_to_ratio all"']
synth_exceptions_cols = ['"similar_to_ratio rand_a" AND "similar_to_ratio rand_b" AND "similar_to_ratio most"']


def test_real():
	res = build_default_results()
	res['page-blocks'] = ([
		'"lenght" AND "height" AND "eccen"',
		'"area" AND "height" AND "lenght"',
		'"blackpix" AND "area" AND "p_black"',
		'"blackpix" AND "mean_tr" AND "wb_trans"',
		'"blackand" AND "area" AND "p_and"'],
		[])
	res['hypothyroid'] = ([], ['"TT4" AND "T4U" AND "FTI"'])
	res['baseball'] = (['"Hits" AND "At_bats" AND "Batting_average"'], 0)
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
