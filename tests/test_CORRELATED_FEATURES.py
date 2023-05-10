import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'CORRELATED_FEATURES'
random.seed(0)

synth_patterns_cols = ['"correlated rand_a" AND "correlated rand_b"']
synth_exceptions_cols = [
	'"correlated rand_a" AND "correlated most"',
	'"correlated rand_b" AND "correlated most"'
]


def test_real():
	res = build_default_results()
	res['blood-transfusion-service-center'] = (['"V2" AND "V3"'], [])
	res['wdbc'] = ([
		'"V1" AND "V3"',
		'"V1" AND "V4"',
		'"V3" AND "V4"',
		'"V21" AND "V24"'
	], [])  # need to debug this case.
	res['ozone-level-8hr'] = ([], [
		'"V28" AND "V29"',
		'"V29" AND "V30"',
		'"V30" AND "V31"',
		'"V31" AND "V32"'
	])
	res['kc2'] = (['"n" AND "v"'], [])    # debug this!!!!
	res['climate-model-simulation-crashes'] = (['"V3" AND "V4"'], [])
	res['madelon'] = ([], [
		'"V65" AND "V337"',
		'"V443" AND "V473"'
	])
	res['nomao'] = (['"V67" AND "V68"'], ['"V68" AND "V70"'])  # The exception here is marginal
	res['mc1'] = ([
		'"CONDITION_COUNT" AND "MULTIPLE_CONDITION_COUNT"',
		'"EDGE_COUNT" AND "NODE_COUNT"',
		'"HALSTEAD_EFFORT" AND "HALSTEAD_PROG_TIME"',
		'"HALSTEAD_EFFORT" AND "HALSTEAD_VOLUME"',
		'"HALSTEAD_PROG_TIME" AND "HALSTEAD_VOLUME"'
	], [
		'"LOC_EXECUTABLE" AND "LOC_TOTAL"', # this i think is wrong, some other as well
	])
	res['pc1'] = ([
		'"loc" AND "lOCode"',
		'"N" AND "V"',
		'"N" AND "total_Op"',
		'"N" AND "total_Opnd"',
		'"V" AND "total_Op"',
		'"V" AND "total_Opnd"',
		'"E" AND "T"',
	], [])  # shows some exceptions, but no red dots
	res['steel-plates-fault'] = ([
		'"V3" AND "V4"',
		'"V5" AND "V22"'], [])
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
