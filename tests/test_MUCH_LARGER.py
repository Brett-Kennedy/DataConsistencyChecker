import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'MUCH_LARGER'
random.seed(0)

synth_patterns_cols = ['"much larger rand" AND "much larger all"']
synth_exceptions_cols = ['"much larger rand" AND "much larger most"']

def test_real():
	res = build_default_results()
	res['vehicle'] = (1, 0)
	res['eucalyptus'] = (8, 16)
	res['gas-drift'] = (0, 588)
	res['adult'] = (1, 0)
	res['anneal'] = (0, ['"thick" AND "width"'])
	res['credit-g'] = (4, 0)
	res['blood-transfusion-service-center'] = (1, 0)
	res['qsar-biodeg'] = (0, 2)
	res['wdbc'] = (132, 11)
	res['diabetes'] = (1, 0)
	res['ozone-level-8hr'] = (80, 80)
	res['climate-model-simulation-crashes'] = (1, ['"V8" AND "V3"', '"V11" AND "V3"', '"V16" AND "V3"'])
	res['ilpd'] = (3, 1)
	res['electricity'] = (0, 5)
	res['MagicTelescope'] = (0, 5)
	res['Click_prediction_small'] = (6, ['"impression" AND "advertiser_id"'])
	res['hypothyroid'] = ([], [
		'"T3" AND "TT4"',
		'"T4U" AND "TT4"',
		'"T3" AND "FTI"'])
	res['baseball'] = (34, 8)
	res['cardiotocography'] = (7, ['"V10" AND "V16"'])
	res['volcanoes-a1'] = (2, 0)
	res[''] = (0, 0)
	res[''] = (0, 0)
	res['wine-quality-white'] = (7, 7)
	res['steel-plates-fault'] = (56, 44)
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
