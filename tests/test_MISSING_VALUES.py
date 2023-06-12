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
	res['bioresponse'] = (1776, [])
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
	res['credit-approval'] = (8, 0)
	res['artificial-characters'] = (7, 0)
	res['splice'] = (60, 0)
	res['har'] = (561, 0)
	res['cmc'] = (9, 0)
	res['segment'] = (18, 0)
	res['JapaneseVowels'] = (14, 0)
	res['jm1'] = (16, 5)
	res['gas-drift'] = (128, 0)
	res['mushroom'] = (20, 0)
	res['irish'] = (3, 0)
	res['profb'] = (7, 0)
	res['adult'] = (11, 0)
	res['higgs'] = (19, 9)
	res['anneal'] = (8, 2)
	res['credit-g'] = (20, 0)
	res['blood-transfusion-service-center'] = (4, 0)
	res['monks-problems-2'] = (6, 0)
	res['tic-tac-toe'] = (9, 0)
	res['qsar-biodeg'] = (41, 0)
	res['wdbc'] = (30, 0)
	res['phoneme'] = (5, 0)
	res['diabetes'] = (8, 0)
	res['ozone-level-8hr'] = (72, 0)
	res['hill-valley'] = (100, 0)
	res['kc2'] = (21, 0)
	res['eeg-eye-state'] = (14, 0)
	res['climate-model-simulation-crashes'] = (20, 0)
	res['spambase'] = (57, 0)
	res['ilpd'] = (10, 0)
	res['one-hundred-plants-margin'] = (64, 0)
	res['banknote-authentication'] = (4, 0)
	res['mozilla4'] = (5, 0)
	res['electricity'] = (8, 0)
	res['madelon'] = (500, 0)
	res['scene'] = (299, 0)
	res['musk'] = (167, 0)
	res['nomao'] = (118, 0)
	res['bank-marketing'] = (16, 0)
	res['MagicTelescope'] = (10, 0)
	res['Click_prediction_small'] = (9, 0)
	res['PhishingWebsites'] = (30, 0)
	res['nursery'] = (8, 0)
	res['page-blocks'] = (10, 0)
	res['hypothyroid'] = (20, ['age'])
	res['yeast'] = (8, 0)
	res['kropt'] = (6, 0)
	res['CreditCardSubset'] = (30, 0)
	res['shuttle'] = (9, 0)
	res['Satellite'] = (36, 0)
	res['baseball'] = (15, 0)
	res['mc1'] = (38, 0)
	res['pc1'] = (21, 0)
	res['cardiotocography'] = (35, 0)
	res['kr-vs-k'] = (6, 0)
	res['volcanoes-a1'] = (3, 0)
	res['wine-quality-white'] = (11, 0)
	res['car-evaluation'] = (21, 0)
	res['solar-flare'] = (12, 0)
	res['allbp'] = (27, 0)
	res['allrep'] = (27, 0)
	res['dis'] = (27, 0)
	res['car'] = (6, 0)
	res['steel-plates-fault'] = (33, 0)
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

