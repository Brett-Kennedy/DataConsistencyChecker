import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'LESS_THAN_ONE'
random.seed(0)

synth_patterns_cols = ['less_than_one all']
synth_exceptions_cols = ['less_than_one most']


def test_real():
	res = build_default_results()
	res['isolet'] = (613, 0)
	res['bioresponse'] = (311, 0)
	res['abalone'] = (3, 2)
	res['cnae-9'] = (0, ['V195', 'V648', 'V731', 'V814'])
	res['SpeedDating'] = (['interests_correlate'], 0)
	res['har'] = (561, 0)
	res['segment'] = (2, 0)
	res['JapaneseVowels'] = (9, 1)
	res['jm1'] = (0, 1)
	res['qsar-biodeg'] = (0, ['V28'])
	res['wdbc'] = (16, 1)
	res['ozone-level-8hr'] = (3, 0)
	res['climate-model-simulation-crashes'] = (17, 0)
	res['spambase'] = (0, ['char_freq_%5B'])
	res['one-hundred-plants-margin'] = (62, 0)
	res['electricity'] = (7, 0)
	res['scene'] = (294, 0)
	res['nomao'] = (89, 0)
	res['MagicTelescope'] = (2, 0)
	res['PhishingWebsites'] = (8, 0)
	res['page-blocks'] = (2, 0)
	res['yeast'] = (6, 0)
	res['baseball'] = (4, 0)
	res['mc1'] = (1, 1)
	res['pc1'] = (0, ['L'])
	res['cardiotocography'] = (['V25'], 0)
	res['volcanoes-a1'] = (['V3'], 0)
	res['wine-quality-white'] = (1, 3)
	res['steel-plates-fault'] = (10, 0)
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
