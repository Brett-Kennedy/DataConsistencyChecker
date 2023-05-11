import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'DECISION_TREE_CLASSIFIER'
random.seed(0)

synth_patterns_cols = ['"dt cls. 1a" AND "dt cls. 1b" AND "dt cls. 3" AND "dt cls. 2"']
synth_exceptions_cols = ['"dt cls. 1b" AND "dt cls. 2" AND "dt cls. 2" AND "dt cls. 3"']


def test_real():
	res = build_default_results()
	res['isolet'] = ([], [
		'"f4" AND "f6" AND "f578"',
		'"f6" AND "f579"'])
	res['bioresponse'] = ([
		'"D3" AND "D28"',
		'"D5" AND "D94"'], [])
	res['soybean'] = (['"precip" AND "temp" AND "sclerotia"'], [])
	res['SpeedDating'] = (9, 10)
	res['credit-approval'] = ([
		'"A3" AND "A5" AND "A4"',
		'"A3" AND "A5"',
		'"A1" AND "A10"'
		], [])
	res['anneal'] = ([], [
		'"carbon" AND "hardness" AND "temper_rolling"',
		'"hardness" AND "formability" AND "len" AND "shape"'
	])
	res[''] = ([], [])
	res[''] = ([], [])
	res[''] = ([], [])

	res['nomao'] = ([
		'"V1" AND "V7"',
		'"V2" AND "V8"'
		], [])
	res['hypothyroid'] = (['"T4U" AND "FTI" AND "FTI_measured"'], [
		'"TT4" AND "T4U" AND "TT4_measured"',
		'"T4U" AND "FTI" AND "T4U_measured"'
	])
	res['mc1'] = (['"BRANCH_COUNT" AND "CYCLOMATIC_COMPLEXITY" AND "MAINTENANCE_SEVERITY"'], [])
	res['car-evaluation'] = (21, [])
	res['allbp'] = ([
		'"TSH" AND "TSH_measured"',
		'"T3" AND "T3_measured"',
		'"TT4" AND "TT4_measured',
		'"T4U" AND "T4U_measured"',
		'"FTI" AND "FTI_measured"',
	], [])
	res['allrep'] = ([
		'"TSH" AND "TSH_measured"',
		'"T3" AND "T3_measured"',
		'"TT4" AND "TT4_measured',
		'"T4U" AND "T4U_measured"',
		'"FTI" AND "FTI_measured"',
	], [])
	res['dis'] = ([
		'"TSH" AND "TSH_measured"',
		'"T3" AND "T3_measured"',
		'"TT4" AND "TT4_measured',
		'"T4U" AND "T4U_measured"',
		'"FTI" AND "FTI_measured"',
	], [])
	res['steel-plates-fault'] = (['"V2" AND "V12"'], [])


	res['hypothyroid'] = (['"T4U" AND "FTI" AND "FTI_measured'],  # still figuring out which are real. this one is good!
		['"TT4" AND "T4U" AND "TT4_measured"',
		 '"T3" AND "T4U" AND "T4U_measured"'])
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
