import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'MATCHED_ZERO'
random.seed(0)

synth_patterns_cols = [
	'"matched zero rand_a" AND "matched zero all"'
]
synth_exceptions_cols = [
	'"matched zero rand_a" AND "matched zero most"',
	'"matched zero all" AND "matched zero most"'
]


def test_real():
	res = build_default_results()
	res['cnae-9'] = (['"V383" AND "V602"'], 0)
	res['segment'] = (16, 50)
	res['jm1'] = (18, 48)
	res['wdbc'] = (15, 0)
	res['kc2'] = (11, 0)
	res['spambase'] = (0, ['"word_freq_857" AND "word_freq_415"'])
	res['nomao'] = (14, 3)
	res['mc1'] = (15, 16)
	res['pc1'] = (16, 6)
	res['steel-plates-fault'] = (['"V1" AND "V15"'], 0)
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
