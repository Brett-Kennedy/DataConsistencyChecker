import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'VERY_LARGE'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = ['very large 3']


def test_real():
	res = build_default_results()
	res['isolet'] = ([], 19)
	res['bioresponse'] = (0, 53)
	res['micro-mass'] = (0, ['V26'])
	res['abalone'] = (0, ['Height'])
	res['vehicle'] = (0, ['SKEWNESS_ABOUT_MAJOR'])
	res['analcatdata_authorship'] = (0,18 )
	res['SpeedDating'] = (0, 4)
	res['eucalyptus'] = (0, ['Rep', 'DBH'])
	res['credit-approval'] = (0, ['A14'])
	res['har'] = (0, 77)
	res['cmc'] = (0, ['Number_of_children_ever_born'])
	res['jm1'] = (0, ['uniq_Op'])
	res['gas-drift'] = (0, 38)
	res['adult'] = (0, ['fnlwgt'])
	res['higgs'] = (0, 4)
	res['credit-g'] = (0, ['duration'])
	res['blood-transfusion-service-center'] = (0, ['V1'])
	res['qsar-biodeg'] = (0, 3)
	res['wdbc'] = (0, 5)
	res['phoneme'] = (0, ['V1'])
	res['ozone-level-8hr'] = (0, 16)
	res['eeg-eye-state'] = (0, ['V2', 'V4', 'V7'])
	res['spambase'] = (0, ['word_freq_you'])
	res['ilpd'] = (0, ['V10'])
	res['one-hundred-plants-margin'] = (0, 12)
	res['electricity'] = (0, 2)
	res['madelon'] = (0, 5)
	res['scene'] = (0, ['attr134'])
	res['musk'] = (0, 4)
	res['MagicTelescope'] = (0, 4)
	res['hypothyroid'] = (0, ['age', 'TT4', 'T4U'])
	res['yeast'] = (0, ['alm', 'mit', 'vac'])
	res['CreditCardSubset'] = (0, 14)
	res['shuttle'] = (0, 6)
	res['baseball'] = (0, 5)
	res['pc1'] = (0, ['uniq_Op'])
	res['cardiotocography'] = (0, 4)
	res['wine-quality-white'] = (0, 8)
	res['steel-plates-fault'] = (0, ['V24'])
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
