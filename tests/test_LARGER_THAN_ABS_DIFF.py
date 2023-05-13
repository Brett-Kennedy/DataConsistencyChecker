import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'LARGER_THAN_ABS_DIFF'
random.seed(0)

synth_patterns_cols = [
	'"larger_diff rand_a" AND "larger_diff rand_b" AND "larger_diff all"',
]
synth_exceptions_cols = [
	'"larger_diff rand_a" AND "larger_diff rand_b" AND "larger_diff most"',
]


def test_real():
	res = build_default_results()
	res['Amazon_employee_access'] = ([], 14)
	res['abalone'] = (0, 16)
	res['vehicle'] = (263, 43)
	res['analcatdata_authorship'] = (22, 142)
	res['SpeedDating'] = (1204, 1141)
	res['eucalyptus'] = (2, 17)
	res['segment'] = (3, 1)
	res['jm1'] = (3, 5)
	res['adult'] = (1, 1)  # This one is weak
	res['higgs'] = ([], 3)
	res['anneal'] = (3, 1)
	res['qsar-biodeg'] = (488, 391)
	res['wdbc'] = (75, 22)
	res['ozone-level-8hr'] = (315, 1940)
	res['kc2'] = (5, 6)
	res['eeg-eye-state'] = (40, 324)
	res['spambase'] = (368, 1667)
	res['electricity'] = ([], ['"nswdemand" AND "vicdemand" AND "transfer"'])
	res['yeast'] = ([], ['"mcg" AND "gvh" AND "alm"', '"mcg" AND "gvh" AND "vac"'])
	res['shuttle'] = ([], [])
	res['baseball'] = ([], [])
	res['mc1'] = ([], [])
	res['pc1'] = ([], [])
	res['cardiotocography'] = ([], [])
	res['wine-quality-white'] = ([], [])
	res['steel-plates-fault'] = ([], [])
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
