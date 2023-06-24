import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SIMILAR_TO_PRODUCT'
random.seed(0)

synth_patterns_cols = ['"similar to prod 1a" AND "similar to prod 1b" AND "similar to prod 2"']
synth_exceptions_cols = ['"similar to prod 1a" AND "similar to prod 1b" AND "similar to prod 3"']


def test_real():
	res = build_default_results()
	res['jm1'] = (0, ['"v" AND "d" AND "e"', '"d" AND "i" AND "v"'])
	res['kc2'] = (0, ['"v" AND "d" AND "e"'])
	res['page-blocks'] = ([
			'"height" AND "lenght" AND "area"',
			'"height" AND "eccen" AND "lenght"',
			'"mean_tr" AND "wb_trans" AND "blackpix"'
		], 0)
	res['mc1'] = (['"HALSTEAD_DIFFICULTY" AND "HALSTEAD_VOLUME" AND "HALSTEAD_EFFORT"'], 0)
	res['pc1'] = (['"V" AND "D" AND "E"'], 0)
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
