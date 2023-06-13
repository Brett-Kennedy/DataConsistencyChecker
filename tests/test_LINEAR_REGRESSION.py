import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'LINEAR_REGRESSION'
random.seed(0)

synth_patterns_cols = [
	'"lin regr 1a" AND "lin regr 1b" AND "lin regr 1c" AND "lin regr 2"'
]
synth_exceptions_cols = [
	'"lin regr 1d" AND "lin regr 1e" AND "lin regr 1f" AND "lin regr 3"'
]

# todo: need to check features on somehow that all features are relevant.
def test_real():
	res = build_default_results()
	res['vehicle'] = ([
		'"SCALED_VARIANCE_MINOR" AND "SCATTER_RATIO"',
		'"SCATTER_RATIO" AND "SCALED_VARIANCE_MINOR"'],
	[])
	res['gas-drift'] = (0, -12)  # Generally good, though flags some points quite close to the predictions
	res['blood-transfusion-service-center'] = (['"V2" AND "V4" AND "V3"'], 0)
	res['kc2'] = (0, ['"e" AND "t"'])
	res['madelon'] = (6, 0)
	res['musk'] = (['"f71" AND "f101" AND "f41"', '"f3" AND "f41" AND "f87" AND "f71"'], 3)
	res['baseball'] = (['"Games_played" AND "At_bats" AND "Runs" AND "Doubles" AND "Triples" AND "RBIs" AND "Hits"'], 0)
	res['mc1'] = (0, ['"HALSTEAD_EFFORT" AND "HALSTEAD_PROG_TIME"'])
	res['pc1'] = (0, ['"loc" AND "N" AND "V" AND "T" AND "E"'])
	res['steel-plates-fault'] = (['"V4" AND "V14" AND "V3"'], 0)
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
		0,
		0)


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
