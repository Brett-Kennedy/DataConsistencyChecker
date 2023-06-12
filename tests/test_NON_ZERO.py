import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'NON_ZERO'
random.seed(0)

synth_patterns_cols = ['non-zero all']
synth_exceptions_cols = ['non-zero most']


def test_real():
	res = build_default_results()
	res['isolet']                   = (70, 397)
	res['bioresponse']              = (28, 31)
	res['mfeat-karhunen']           = (64, 0)
	res['Amazon_employee_access']   = (8, ['RESOURCE'])
	res['abalone']                  = (6, ['Height'])
	res['vehicle']                  = (16, 0)
	res['satimage']                 = (36, 0)
	res['analcatdata_authorship']   = (12, 7)
	res['breast-w']                 = (9, 0)
	res['SpeedDating'] = (18, 24)
	res['eucalyptus'] = (9, 3)
	res['vowel'] = (6, 4)
	res['wall-robot-navigation'] = (24, 0)
	res['credit-approval'] = (1, 0)
	res['artificial-characters'] = (2, 0)
	res['splice'] = (0, 0)
	res['har'] = (552, 6)
	res['JapaneseVowels'] = (13, 1)  # In this case, the 0 is simply another value in a distribution, and we may wish to not flag these
	res['gas-drift'] = (122, 6)
	res['anneal'] = (2, ['width'])
	res['qsar-biodeg'] = (12, ['V8'])
	res['ozone-level-8hr'] = (40, 20)
	res['electricity'] = (1, 6)
	res['madelon'] = (499, 1)
	res['scene'] = (10, 122)
	res['musk'] = (18, 122)
	res['nomao'] = (45, 28)
	res['bank-marketing'] = (4, ['V12'])
	res['MagicTelescope'] = (5, 4)
	res['yeast'] = (3, ['mit', 'vac', 'nuc'])
	res['baseball'] = (12, ['Triples', 'Home_runs', 'Strikeouts'])
	res['mc1'] = (7, ['CYCLOMATIC_DENSITY', 'NORMALIZED_CYLOMATIC_COMPLEXITY'])
	res['wine-quality-white'] = (10, ['V3'])
	res['allbp'] = (0, 5)
	res['allrep'] = (0, 5)
	res['dis'] = (0, 5)
	res['steel-plates-fault'] = (17, 4)
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
