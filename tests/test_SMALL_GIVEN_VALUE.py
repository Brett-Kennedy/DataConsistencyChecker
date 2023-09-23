import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SMALL_GIVEN_VALUE'
random.seed(0)

synth_patterns_cols = []  # None for this test
synth_exceptions_cols = ['"small_given rand" AND "small_given most"',
                         '"small_given rand" AND "small_given date_most"']


def test_real():
	res = build_default_results()
	res['isolet'] = (0, 19)
	res['qsar-biodeg'] = ([], [
		'"V25" AND "V1"',
		'"V25" AND "V27"'])
	res['scene'] = ([], [
		'"Beach" AND "attr29"',
		'"Beach" AND "attr30"',
		'"Beach" AND "attr31"',
		'"Beach" AND "attr32"',
		'"Beach" AND "attr34"',
		'"Beach" AND "attr36"',
		'"Beach" AND "attr37"',
		'"Beach" AND "attr38"',
		'"Beach" AND "attr39"',
		'"Beach" AND "attr40"',
		'"Beach" AND "attr44"',
		'"Beach" AND "attr45"',
		'"Beach" AND "attr46"',
		'"Beach" AND "attr47"',
		'"Beach" AND "attr146"',
		'"FallFoliage" AND "attr211"',
		'"FallFoliage" AND "attr224"',
		'"FallFoliage" AND "attr226"',
		'"FallFoliage" AND "attr227"',
		'"FallFoliage" AND "attr230"',
		'"Field" AND "attr29"',
		'"Field" AND "attr40"',
		'"Field" AND "attr41"',
		'"Field" AND "attr225"',
		'"Field" AND "attr226"',
		'"Field" AND "attr232"',
		'"Field" AND "attr233"',
		'"Field" AND "attr234"',
		'"Field" AND "attr235"',
		'"Field" AND "attr241"',
		'"Field" AND "attr242"',
		'"Field" AND "attr243"',
		'"Field" AND "attr244"',
		])
	res['hypothyroid'] = ([], ['"T4U_measured" AND "T3"', '"FTI_measured" AND "T3"'])
	res['allbp'] = ([], [
		'"TT4_measured" AND "TSH"',
		'"TT4_measured" AND "T4U"',
		'"TT4_measured" AND "FTI"',
		'"T4U_measured" AND "FTI"',
		])
	res['allrep'] = ([], [
		'"TT4_measured" AND "TSH"',
		'"TT4_measured" AND "T4U"',
		'"TT4_measured" AND "FTI"',
		'"T4U_measured" AND "FTI"'
	])
	res['dis'] = ([], [
		'"TT4_measured" AND "TSH"',
		'"TT4_measured" AND "T4U"',
		'"TT4_measured" AND "FTI"',
		'"T4U_measured" AND "FTI"',
	])
	res['steel-plates-fault'] = ([], [
		'"V28" AND "V25"',
		'"V30" AND "V10"',
		'"V30" AND "V16"',
		'"V30" AND "V22"',
		'"V30" AND "V24"',
		'"V33" AND "V9"',
		'"V33" AND "V20"',
	])
	real_test(test_id, res) # steel-plates-fault has 1 good one


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
