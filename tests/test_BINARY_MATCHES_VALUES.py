import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'BINARY_MATCHES_VALUES'
random.seed(0)

synth_patterns_cols = ['"bin_match_val rand_a" AND "bin_match_val all"']
synth_exceptions_cols = ['"bin_match_val rand_a" AND "bin_match_val most"']


def test_real():
	res = build_default_results()
	res['credit-approval'] = (['"A10" AND "A11"'], [])
	res['nomao'] = ([
		'"V7" AND "V1"',
		'"V8" AND "V2"',
		'"V8" AND "V4"'
		], [
		'"V8" AND "V3"',
		'"V8" AND "V6"'])
	res['allbp'] = ([
		'"FTI" AND "FTI_measured"'
		], [
		'"T3" AND "T3_measured"',
		'"T4U" AND "T4U_measured"',
		'"FTI" AND "T4U_measured"',
		'"T4U" AND "FTI_measured"'])
	res['allrep'] = ([
		'"FTI" AND "FTI_measured"'
		], [
		'"T3" AND "T3_measured"',
		'"T4U" AND "T4U_measured"',
		'"FTI" AND "T4U_measured"',
		'"T4U" AND "FTI_measured"'])
	res['dis'] = ([
		'"FTI" AND "FTI_measured"'
		], [
		'"T3" AND "T3_measured"',
		'"T4U" AND "T4U_measured"',
		'"FTI" AND "T4U_measured"',
		'"T4U" AND "FTI_measured"'])
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

