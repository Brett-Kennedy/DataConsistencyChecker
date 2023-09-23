import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'NUMBER_DECIMALS'
random.seed(0)

synth_patterns_cols = ['num_decimals all']
synth_exceptions_cols = ['num_decimals rand', 'num_decimals most']


def test_real():
	res = build_default_results()
	res['isolet'] = (-540, 0)
	res['bioresponse'] = (-75, -130)
	res['mfeat-karhunen'] = (64, 0)
	res['abalone'] = (7, 0)
	res['satimage'] = (36, 0)
	res['SpeedDating'] = (1, 0)
	res['eucalyptus'] = (2, 0)
	res['vowel'] = (10, 0)
	res['wall-robot-navigation'] = (24, 0)
	res['credit-approval'] = (3, 0)
	res['artificial-characters'] = (2, 0)
	res['har'] = (-540, 0)
	res['segment'] = (2, [
		'hue-mean'
	])
	res['JapaneseVowels'] = (12, 0)
	res['jm1'] = (7, 0)
	res['gas-drift'] = (128, 0)
	res['higgs'] = (-19, [
		'lepton_eta',
		'missing_energy_phi',
		'jet1eta',
		'jet1phi',
		'jet2eta'
	])
	res['qsar-biodeg'] = (15, 0)
	res['wdbc'] = (0, [
		'V28'
	])
	res['phoneme'] = (5, 0)
	res['diabetes'] = (2, 0)
	res['hill-valley'] = (100, 0)
	res['kc2'] = (7, 0)
	res['eeg-eye-state'] = (13, [
		'V13'
	])
	res['climate-model-simulation-crashes'] = (17, 0)
	res['spambase'] = (2, 0)
	res['ilpd'] = (4, 0)
	res['one-hundred-plants-margin'] = (55, 0)
	res['electricity'] = (7, 0)
	res['scene'] = (294, 0)
	res['nomao'] = (55, 0)
	res['MagicTelescope'] = (10, 0)
	res['page-blocks'] = (4, 0)
	res['hypothyroid'] = (2, 0)
	res['yeast'] = (6, 0)
	res['CreditCardSubset'] = (29, 0)
	res['baseball'] = (4, 0)
	res['mc1'] = (8, 0)
	res['pc1'] = (7, 0)
	res['cardiotocography'] = (2, 0)
	res['volcanoes-a1'] = (1, 0)
	res['wine-quality-white'] = (8, 0)
	res['steel-plates-fault'] = (12, 0)
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
		2,
		1)


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
		2,
		0)
