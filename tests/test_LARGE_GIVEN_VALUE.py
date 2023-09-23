import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'LARGE_GIVEN_VALUE'
random.seed(0)

synth_patterns_cols = []  # None for this test
synth_exceptions_cols = [
	'"large_given rand" AND "large_given most"',
	'"large_given rand" AND "large_given date_most"'
]


def test_real():
	res = build_default_results()
	res['isolet'] = ([], 24)
	res['abalone'] = ([], ['"Sex" AND "Whole_weight"', '"Sex" AND "Viscera_weight"', '"Sex" AND "Shell_weight"'])
	res['SpeedDating'] = ([], 26)
	res['eucalyptus'] = ([], ['"Locality" AND "DBH"', '"Frosts" AND "DBH"'])
	res['credit-approval'] = ([], 6)
	res['adult'] = ([], 4)
	res['credit-g'] = ([], [
		'"property_magnitude" AND "duration"',
		'"property_magnitude" AND "credit_amount"',
		'"job" AND "credit_amount"'])
	res['qsar-biodeg'] = ([], [
		'"V25" AND "V9"',
		'"V25" AND "V14"',
		'"V25" AND "V35"'
	])
	res['scene'] = ([], 97)  # As there are many, we simply provide the count
	res['bank-marketing'] = ([], ['"V3" AND "V1"'])
	res['hypothyroid'] = ([], [
		'"referral_source" AND "TT4"',
		'"referral_source" AND "FTI"',
		'"sex" AND "TT4"',
		'"sex" AND "T4U"',
		'"sick" AND "T3"',
		'"sick" AND "FTI"',
	])
	res['baseball'] = ([], [
		'"Position" AND "Number_seasons"',
		'"Position" AND "Triples"',
	])
	res['mc1'] = ([], [
		'"ESSENTIAL_DENSITY" AND "CYCLOMATIC_DENSITY"',
		'"ESSENTIAL_DENSITY" AND "NORMALIZED_CYLOMATIC_COMPLEXITY"',
		# '"GLOBAL_DATA_DENSITY" AND "HALSTEAD_LEVEL"', -- no longer appearing due to tighter definition
	])
	res['cardiotocography'] = ([], 13)
	res['allbp'] = ([], [
		'"psych" AND "T3"',
		'"psych" AND "T4U"',
		'"T3_measured" AND "T3"',
		'"T4U_measured" AND "T4U"',
		'"FTI_measured" AND "T4U"',
	])
	res['allrep'] = ([], [
		'"psych" AND "T3"',
		'"psych" AND "T4U"',
		'"T3_measured" AND "T3"',
		'"T4U_measured" AND "T4U"',
		'"FTI_measured" AND "T4U"',
	])
	res['dis'] = ([], [
		'"psych" AND "T3"',
		'"psych" AND "T4U"',
		'"T3_measured" AND "T3"',
		'"T4U_measured" AND "T4U"',
		'"FTI_measured" AND "T4U"',
	])
	res['steel-plates-fault'] = ([], [
		'"V12" AND "V22"',
		'"V12" AND "V23"',
		'"V13" AND "V22"',
		'"V13" AND "V23"',
		'"V28" AND "V18"',
		'"V29" AND "V11"',
		'"V30" AND "V14"',
		'"V30" AND "V22"',
		'"V30" AND "V23"',
		'"V33" AND "V22"',
		'"V33" AND "V23"',
	])
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
		0,
		0)


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
