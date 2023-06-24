import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'SIMILAR_WRT_RATIO'
random.seed(0)

synth_patterns_cols = [
	'"sim wrt ratio rand_a" AND "sim wrt ratio all"'
]
synth_exceptions_cols = [
	'"sim wrt ratio rand_a" AND "sim wrt ratio most"'
]


def test_real():
	res = build_default_results()
	res['Amazon_employee_access'] = (0, ['"ROLE_ROLLUP_2" AND "ROLE_CODE"'])
	res['abalone'] = (0, ['"Length" AND "Diameter"'])
	res['vehicle'] = (20, 6)
	res['SpeedDating'] = (0, ['"age" AND "age_o"'])
	res['jm1'] = (0, ['"v(g)" AND "branchCount"'])
	res['gas-drift'] = (7, 126)
	res['higgs'] = (0, ['"m_wbb" AND "m_wwbb"'])
	res['qsar-biodeg'] = (4, ['"V1" AND "V13"'])
	res['wdbc'] = (3, 3)
	res['ozone-level-8hr'] = (4, 51)
	res['hill-valley'] = (2596, 2354)
	res['eeg-eye-state'] = (3, 88)
	res['nomao'] = (2, 27)
	res['hypothyroid'] = ([], ['"TT4" AND "FTI"'])
	res['baseball'] = ([
		'"Batting_average" AND "On_base_pct"',
		'"On_base_pct" AND "Slugging_pct"'
	    ], [
		'"Batting_average" AND "Slugging_pct"'
	])
	res['mc1'] = (0, ['"DECISION_COUNT" AND "MULTIPLE_CONDITION_COUNT"'])
	res['pc1'] = (0, ['"loc" AND "lOCode"', '"v(g)" AND "branchCount"'])
	res['cardiotocography'] = (8, 4)
	res['steel-plates-fault'] = (['"V3" AND "V4"'], 0)
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
