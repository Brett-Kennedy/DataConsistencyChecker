import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'GREATER_THAN_ONE'
random.seed(0)

synth_patterns_cols = ['greater_than_one all']
synth_exceptions_cols = ['greater_than_one most']


def test_real():
	res = build_default_results()
	res['bioresponse'] = ([], 127)
	res['SpeedDating'] = (57, ['expected_num_matches'])
	res['eucalyptus'] = (8, 5)
	res['vowel'] = ([], ['Feature_0'])
	res['wall-robot-navigation'] = (3, ['V4'])
	res['gas-drift'] = (6, 44)
	res['qsar-biodeg'] = (31, ['V2'])
	res['ozone-level-8hr'] = (8, [
		'V25', 'V27', 'V36', 'V37', 'V38', 'V39', 'V40', 'V43', 'V44', 'V45', 'V46', 'V47',
	    'V48', 'V49', 'V50', 'V52', 'V69'])
	res['hill-valley'] = ([], 35)
	res['ilpd'] = (5, ['V9'])
	res['MagicTelescope'] = (3, ['fWidth', 'fM3Long'])
	res['mc1'] = (27, ['HALSTEAD_DIFFICULTY', 'PERCENT_COMMENTS'])
	res['PC1'] = (17, ['D'])
	res['steel-plates-fault'] = (12, ['V22'])
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
