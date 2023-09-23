import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'LARGER'
random.seed(0)

synth_patterns_cols = ['"larger rand" AND "larger all" AND "larger most" AND "larger all_2"']
synth_exceptions_cols = ['"larger most" AND "larger rand"']


# !!!! I've updated check_larger since this to exclude cases where the 2 columns are almost the same.
def test_real():
	res = build_default_results()
	res['abalone'] = (2, 8)
	res['cnae-9'] = (5, -70)
	res['vehicle'] = (80, 6)
	res['analcatdata_authorship'] = (48, 152)
	res['SpeedDating'] = (96, 78)
	res['eucalyptus'] = (2, 15)
	res['vowel'] = (4, [])
	res['credit-approval'] = ([], [
		'"A2" AND "A3"',
		'"A2" AND "A11"'
	])
	res['artificial-characters'] = (5, [])
	res['cmc'] = (1, [])
	res['segment'] = (25, 14)
	res['JapaneseVowels'] = (0, ['"frame" AND "coefficient3'])
	res['jm1'] = (10, 70)
	res['gas-drift'] = (2330, 2594)
	res['irish'] = (1, [])
	res['profb'] = (3, [])
	res['adult'] = (2, 3)
	res['anneal'] = (5, 2)
	res['credit-g'] = (2, [])
	res['qsar-biodeg'] = (125, 72)
	res['wdbc'] = (57, 17)
	res['ozone-level-8hr'] = (89, 209)
	res['kc2'] = (7, 52)
	res['eeg-eye-state'] = ([], 75)
	res['climate-model-simulation-crashes'] = (17, ['"V3" AND "V1"'])
	res['spambase'] = (40, 92)
	res['ilpd'] = (6, [
		'"V1" AND "V4"',
		'"V6" AND "V3"',
		'"V7" AND "V3"'
	])
	res['one-hundred-plants-margin'] = ([], 25)
	res['mozilla4'] = (1, ['"end" AND "size"'])
	res['electricity'] = (5, 7)
	res['scene'] = ([], 24)
	res['musk'] = (50, 122)
	res['nomao'] = (-2400, -300)
	res['bank-marketing'] = ([], ['"V1" AND "V13"', '"V1" AND "V15"', '"V12" AND "V15"'])
	res['MagicTelescope'] = (3, 4)
	res['Click_prediction_small'] = (1, 3)
	res['page-blocks'] = (13, ['"blackpix" AND "height"'])
	res['hypothyroid'] = ([], 2)
	res['CreditCardSubset'] = ([], 2)
	res['shuttle'] = ([], [])
	res['baseball'] = ([], [])
	res['mc1'] = ([], [])
	res['pc1'] = ([], [])
	res['cardiotocography'] = ([], [])
	res['wine-quality-white'] = ([], [])
	res['allbp'] = ([], [])
	res['allrep'] = ([], [])
	res['dis'] = ([], [])
	res['steel-plates-fault'] = ([], [])
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
