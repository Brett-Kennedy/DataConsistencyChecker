import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'MISSING_VALUES'
random.seed(0)

synth_patterns_cols = ['missing vals all']
synth_exceptions_cols = ['missing vals most', 'missing vals most null']


def test_real():
	res = build_default_results()
	res['isolet'] = (617, [])
	res['soybean'] = (1, 2)
	res['micro-mass'] = (1087, 0)
	res['mfeat-karhunen'] = (64, 0)
	res['Amazon_employee_access'] = (9, 0)
	res['abalone'] = (8, 0)
	res['cnae-9'] = (856, 0)
	res['semeion'] = (256, 0)
	res['vehicle'] = (18, 0)
	res['satimage'] = (36, 0)
	res['analcatdata_authorship'] = (70, 0)
	res['breast-w'] = (8, 0)
	res['SpeedDating'] = (60, 0)
	res['eucalyptus'] = (10, 2)
	res['vowel'] = (13, 0)
	res['wall-robot-navigation'] = (24, 0)
	res['credit-approval'] = (0, 0)
	res['artificial-characters'] = (0, 0)
	res['splice'] = (0, 0)
	res['har'] = (0, 0)
	res['cmc'] = (0, 0)
	res['segment'] = (0, 0)
	res['JapaneseVowels'] = (0, 0)
	res['jm1'] = (0, 0)
	res['gas-drift'] = (0, 0)
	res['mushroom'] = (0, 0)
	res['irish'] = (0, 0)
	res['profb'] = (0, 0)
	res['adult'] = (0, 0)
	res['higgs'] = (0, 0)
	res['anneal'] = (0, 0)
	res['credit-g'] = (0, 0)
	res['blood-transfusion-service-center'] = (0, 0)
	res['monks-problems-2'] = (0, 0)
	res['tic-tac-toe'] = (0, 0)
	res['qsar-biodeg'] = (0, 0)
	res['wdbc'] = (0, 0)
	res['phoneme'] = (0, 0)
	res['diabetes'] = (0, 0)
	res['ozone-level-8hr'] = (0, 0)
	res['hill-valley'] = (0, 0)
	res['kc2'] = (0, 0)
	res['eeg-eye-state'] = (0, 0)
	res['climate-model-simulation-crashes'] = (0, 0)
	res['spambase'] = (0, 0)
	res['ilpd'] = (0, 0)
	res['one-hundred-plants-margin'] = (0, 0)
	res['banknote-authentication'] = (0, 0)
	res['mozilla4'] = (0, 0)
	res['electricity'] = (0, 0)
	res['madelon'] = (0, 0)
	res['scene'] = (0, 0)
	res['musk'] = (0, 0)
	res['nomao'] = (0, 0)
	res['bank-marketing'] = (0, 0)
	res['MagicTelescope'] = (0, 0)
	res['Click_prediction_small'] = (0, 0)
	res['PhishingWebsites'] = (0, 0)
	res['nursery'] = (0, 0)
	res['page-blocks'] = (0, 0)
	res['hypothyroid'] = ([], ['age'])
	res['yeast'] = (0, 0)
	res['kropt'] = (0, 0)
	res['CreditCardSubset'] = (0, 0)
	res['shuttle'] = (0, 0)
	res['Satellite'] = (0, 0)
	res['baseball'] = (0, 0)
	res['mc1'] = (0, 0)
	res['pc1'] = (0, 0)
	res['cardiotocography'] = (0, 0)
	res['kr-vs-k'] = (0, 0)
	res['volcanoes-a1'] = (0, 0)
	res['wine-quality-white'] = (0, 0)
	res['car-evaluation'] = (0, 0)
	res['solar-flare'] = (0, 0)
	res['allbp'] = (0, 0)
	res['allrep'] = (0, 0)
	res['dis'] = (0, 0)
	res['car'] = (0, 0)
	res['steel-plates-fault'] = (0, 0)
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
		[],
		['missing vals all', 'missing vals most', 'missing vals most null'])


def test_synthetic_in_sync_nulls():
	synth_test(
		test_id,
		'in-sync',
		[],
		['missing vals most null'])  # Mostly NUll, with some non-null


def test_synthetic_random_nulls():
	synth_test(
		test_id,
		'random',
		[],
		['missing vals most null'])


def test_synthetic_80_percent_nulls():
	synth_test(
		test_id,
		'80-percent',
		[],
		['missing vals most null'])


def test_synthetic_all_cols_no_nulls():
	synth_test_all_cols(
		test_id,
		'none',
		['missing vals all'],
		['missing vals most', 'missing vals most null'])


def test_synthetic_all_cols_one_row_nulls():
	synth_test_all_cols(
		test_id,
		'one-row',
		[],
		['missing vals all', 'missing vals most', 'missing vals most null'])


def test_synthetic_all_cols_in_sync_nulls():
	synth_test_all_cols(
		test_id,
		'in-sync',
		[],
		['missing vals most null'])  # Mostly NUll, with some non-null


def test_synthetic_all_cols_random_nulls():
	synth_test_all_cols(
		test_id,
		'random',
		[],
		['missing vals most null'])


def test_synthetic_all_cols_80_percent_nulls():
	synth_test_all_cols(
		test_id,
		'80-percent',
		[],
		['missing vals most null'])

