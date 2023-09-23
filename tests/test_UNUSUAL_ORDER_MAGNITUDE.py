import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'UNUSUAL_ORDER_MAGNITUDE'
random.seed(0)

synth_patterns_cols = ['unusual_number all']
synth_exceptions_cols = ['unusual_number most_1', 'unusual_number most_2']


def test_real():
	res = build_default_results()
	res['Amazon_employee_access'] = (2, ['MGR_ID', 'ROLE_ROLLUP_2', 'ROLE_FAMILY_DESC'])
	res['vehicle'] = (10, ['SCALED_VARIANCE_MINOR', 'SKEWNESS_ABOUT_MAJOR'])
	res['analcatdata_authorship'] = (0, ['and', 'of', 'to'])
	res['eucalyptus'] = (1, ['Rep', 'PMCno', 'Ht'])
	res['wall-robot-navigation'] = (3, 0)
	res['artificial-characters'] = (1, 0)
	res['cmc'] = (5, 0)
	res['jm1'] = (0, 5)
	res['profb'] = (1, 0)
	res['adult'] = (0, ['fnlwgt'])
	res['credit-g'] = (4, 0)
	res['monks-problems-2'] = (4, 0)
	res['qsar-biodeg'] = (5, 0)
	res['wdbc'] = (1, ['V2'])
	res['diabetes'] = (1, 0)
	res['ozone-level-8hr'] = (3, ['V70'])
	res['climate-model-simulation-crashes'] = (1, 0)
	res['kc2'] = (0, 5)
	res['eeg-eye-state'] = (7, 7)
	res['spambase'] = (0, 3)
	res['ilpd'] = (1, 0)
	res['mozilla4'] = (0, ['id', 'size'])
	res['electricity'] = (1,0)
	res['madelon'] = (498, 1)
	res['musk'] = (0, 10)
	res['nomao'] = (27, 0)
	res['bank-marketing'] = (1, 0)
	res['MagicTelescope'] = (1, 2)
	res['Click_prediction_small'] = (2, 2)
	res['page_blocks'] = ([], ['mean_tr', 'blackpix', 'wb_trans'])
	res['hypothyroid'] = ([], ['age', 'TT4', 'FTI'])
	res['kropt'] = (3, 0)
	res['CreditCardSubset'] = (0, ['Time'])
	res['shuttle'] = (0, ['A7, A8'])
	res['Satellite'] = (0, 9)
	res['baseball'] = (1, 2)
	res['mc1'] = (0, 5)
	res['pc1'] = (0, 4)
	res['cardiotocography'] = (4, 0)
	res['kr-vs-k'] = (6, 0)
	res['wine-quality-white'] = (1, 3)
	res['solar-flare'] = (1, 0)
	res['steel-plates-fault'] = (1, 5)
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
