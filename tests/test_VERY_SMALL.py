import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'VERY_SMALL'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = ['very small most']


def test_real():
	res = build_default_results()
	res['isolet'] = ([], 45)
	res['bioresponse'] = (0, 12)
	res['mfeat-karhunen'] = (0, 10)
	res['Amazon_employee_access'] = (0, ['ROLE_ROLLUP_2'])
	res['SpeedDating'] = (0, 4)
	res['eucalyptus'] = (0, ['Ins_res', 'Stem_Fm', 'Crown_Fm'])
	res['har'] = (0, 65)
	res['JapaneseVowels'] = (0, ['coefficient11'])
	res['gas-drift'] = (0, 4)
	res['adult'] = (0, ['education-num'])
	res['higgs'] = (0, ['m_lv'])
	res['qsar-biodeg'] = (0, 7)
	res['ozone-level-8hr'] = (0, 11)
	res['eeg-eye-state'] = (0, 12)
	res['madelon'] = (0, 144)
	res['scene'] = (0, 8)
	res['musk'] = (0, 6)
	res['nomao'] = (0, 10)
	res['bank-marketing'] = (0, ['V6'])
	res['page-blocks'] = ([], ['p_and'])
	res['hypothyroid'] = ([], ['T4U', 'FTI'])
	res['yeast'] = (0, ['nuc'])
	res['CreditCardSubset'] = (0, 15)
	res['shuttle'] = (0, 6)
	res['Satellite'] = (0, 9)
	res['baseball'] = (0, ['Batting_average', 'On_base_pct'])
	res['cardiotocography'] = (0, ['V22', 'V23'])
	res['wine-quality-white'] = (0, ['V1'])
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
