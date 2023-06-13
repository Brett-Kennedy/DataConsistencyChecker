import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'RARE_COMBINATION'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = ['"rare_combo all_a" AND "rare_combo all_b"']


def test_real():
	res = build_default_results()
	res['analcatdata_authorship'] = (0, 66)
	res['breast-w'] = (0, ['"Bare_Nuclei" AND "Mitoses"'])
	res['SpeedDating'] = (0, 12)
	res['wall-robot-navigation'] = (0, 8)
	res['gas-drift'] = (0, 33)
	res['qsar-biodeg'] = (0, 7)
	res['wdbc'] = (0, 10)
	res['spambase'] = (0, [
		'"word_freq_business" AND "word_freq_labs"',
		'"word_freq_your" AND "word_freq_hp"',
		'"word_freq_your" AND "word_freq_original"'
	])
	res['ilpd'] = (0, [
		'"V4" AND "V5"',
		'"V5" AND "V6"',
		'"V5" AND "V9"'
	])
	res['one-hundred-plants-margin'] = (0, 14)
	res['scene'] = (0, 1079)
	res['musk'] = (0, 56)
	res['nomao'] = (0, 391)
	res['Click_prediction_small'] = (0, [
		'"title_id" AND "user_id"',
		'"description_id" AND "user_id"'
	])
	res['cardiotocography'] = (0, ['"V8" AND "V11"'])
	res['steel-plates-fault'] = (0, ['"V3" AND "V16"', '"V3" AND "V19"', '"V4" AND "V16"', '"V4" AND "V19"'])

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
		0,
		0)


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
