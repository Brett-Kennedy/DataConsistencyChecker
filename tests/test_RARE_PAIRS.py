import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'RARE_PAIRS'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = ['"rare_pair rand_a" AND "rare_pair rand_b"']


def test_real():
	res = build_default_results()
	res['soybean'] = (0, ['"leaf-malf" AND "seed-discolor"'])
	res['semeion'] = (0, 39)
	res['qsar-biodeg'] = (0, ['"V24" AND "V25"', '"V25" AND "V29"'])
	res['scene'] = (0, ['"Beach" AND "Field"'])
	res['hypothyroid'] = ([], ['"T4U_measured" AND "FTI_measured"'])
	res['solar-flare'] = (0, [
		'"Historically-complex" AND "Area"',
		'"Did_region_become_historically_complex" AND "Area_of_the_largest_spot"'])
	res['allbp'] = ([], ['"T4U_measured" AND "FTI_measured"'])
	res['allrep'] = ([], ['"T4U_measured" AND "FTI_measured"'])
	res['dis'] = ([], ['"T4U_measured" AND "FTI_measured"'])
	res['steel-plates-fault'] = (0, [
		'"V12" AND "V30"',
		'"V13" AND "V30"'
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
