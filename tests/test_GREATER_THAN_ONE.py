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
	res['micro-mass'] = (1045, 0)
	res['Amazon_employee_access'] = (9, 0)
	res['cnae-9'] = (97, 0)
	res['vehicle'] = (18, 0)
	res['analcatdata_authorship'] = (70, 0)
	res['breast-w'] = (9, 0)
	res['SpeedDating'] = (57, ['expected_num_matches'])
	res['eucalyptus'] = (8, 5)
	res['vowel'] = ([], ['Feature_0'])
	res['wall-robot-navigation'] = (3, ['V4'])
	res['artificial-characters'] = (7, 0)
	res['cmc'] = (6, 0)
	res['segment'] = (2, 0)
	res['JapaneseVowels'] = (2, 0)
	res['jm1'] = (18, 0)
	res['gas-drift'] = (6, 44)
	res['irish'] = (2, 0)
	res['profb'] = (5, 0)
	res['adult'] = (6, 0)
	res['higgs'] = (4, 0)
	res['anneal'] = (7, 0)
	res['credit-g'] = (6, 0)
	res['blood-transfusion-service-center'] = (4, 0)
	res['monks-problems-2'] = (4, 0)
	res['qsar-biodeg'] = (31, ['V2'])
	res['wdbc'] = (9, 0)
	res['diabetes'] = (7, 0)
	res['ozone-level-8hr'] = (8, [
		'V25', 'V27', 'V36', 'V37', 'V38', 'V39', 'V40', 'V43', 'V44', 'V45', 'V46', 'V47',
	    'V48', 'V49', 'V50', 'V52', 'V69'])
	res['hill-valley'] = ([], 35)
	res['kc2'] = (2, 0)
	res['eeg-eye-state'] = (14, 0)
	res['climate-model-simulation-crashes'] = (3, 0)
	res['spambase'] = (3, 0)
	res['ilpd'] = (5, ['V9'])
	res['mozilla4'] = (4, 0)
	res['electricity'] = (1, 0)
	res['madelon'] = (500, 0)
	res['musk'] = (166, 0)
	res['nomao'] = (27, 0)
	res['bank-marketing'] = (7, 0)
	res['MagicTelescope'] = (3, ['fWidth:', 'fM3Long:'])
	res['Click_prediction_small'] = (9, 0)
	res['PhishingWebsites'] = (8, 0)
	res['page-blocks'] = (7, 0)
	res['hypothyroid'] = (3, 0)
	res['kropt'] = (3, 0)
	res['CreditCardSubset'] = (1, 0)
	res['shuttle'] = (9, 0)
	res['Satellite'] = (36, 0)
	res['baseball'] = (11, 0)
	res['mc1'] = (27, ['HALSTEAD_DIFFICULTY', 'PERCENT_COMMENTS'])
	res['PC1'] = (17, ['D'])
	res['cardiotocography'] = (22, 0)
	res['kr-vs-k'] = (6, 0)
	res['volcanoes-a1'] = (2, 0)
	res['wine-quality-white'] = (5, 0)
	res['solar-flare'] = (3, 0)
	res['allbp'] = (8, 0)
	res['allrep'] = (8, 0)
	res['dis'] = (8, 0)
	res[''] = (0, 0)

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
