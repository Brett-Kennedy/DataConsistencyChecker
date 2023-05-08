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
	res['qsar-biodeg'] = ([], ['V2'])
	res['ozone-level-8hr'] = ([], ['V25', 'V27', 'V36', 'V37', 'V38', 'V39', 'V40', 'V43', 'V44', 'V45', 'V46', 'V47',
	                               'V48', 'V49', 'V50', 'V52', 'V69'])
	res['hill-valley'] = ([], ['V1', 'V6', 'V10', 'V13', 'V15', 'V18', 'V19', 'V21', 'V24', 'V25', 'V26', 'V27', 'V28',
	                           'V30', 'V32', 'V36', 'V40', 'V41', 'V45', 'V48', 'V49', 'V60', 'V62', 'V70', 'V71',
	                           'V78', 'V79', 'V81', 'V83', 'V89', 'V90', 'V93', 'V96', 'V97', 'V99', 'V100'])
	res['ilpd'] = ([], ['V9'])
	res['MagicTelescope'] = ([], ['fWidth', 'fM3Long'])
	res['mc1'] = ([], 'HALSTEAD_DIFFICULTY', 'PERCENT_COMMENTS'])
	res['PC1'] = ([], ['D'])
	res['steel-plates-fault'] = ([], ['V22'])
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
