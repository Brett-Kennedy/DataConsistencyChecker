import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'FEW_WITHIN_RANGE'
random.seed(0)

synth_patterns_cols = ['few in range all']
synth_exceptions_cols = ['few in range most', 'few in range date_most']


def test_real():
	res = build_default_results()
	res['bioresponse'] = (7, 8)
	res['micro-mass'] = (2, [])
	res['Amazon_employee_access'] = ([], ['ROLE_FAMILY'])
	res['wall-robot-navigation'] = (2, 7)
	res['har'] = ([], ['V559'])
	res['segment'] = (2, ['hue-mean'])
	res['electricity'] = (1, [])
	res['musk'] = (2, ['f61', 'f104', 'f105'])
	res['nomao'] = (2, ['V89', 'V103', 'V107'])
	res['Click_prediction_small'] = (1, [])
	res['mc1'] = ([], ['CYCLOMATIC_DENSITY'])
	res['allbp'] = ([], ['T3'])
	res['allrep'] = ([], ['T3'])
	res['dis'] = ([], ['T3'])
	res['steel-plates-fault'] = ([], ['V11'])
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
		0)  # It is random that 'random' still identifies patt


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
