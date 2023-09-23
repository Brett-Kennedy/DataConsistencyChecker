import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'BINARY_NUM_SAME'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = [
	'"bin_num_same rand_a" AND "bin_num_same rand_b" AND "bin_num_same rand_c" AND "bin_num_same rand_d" AND "bin_num_same most"'
]


def test_real():
	res = build_default_results()
	res['isolet'] = ([], ['"f578" AND "f579" AND "f580"'])
	res['cardiotocography'] = ([], ['"V14" AND "V26" AND "V27" AND "V28" AND "V29" AND "V30" AND "V31" AND "V32" AND "V33" AND "V34" AND "V35"'])
	res['car-evaluation'] = (['"buying_price_vhigh" AND "buying_price_high" AND "buying_price_med" AND "buying_price_low" AND "maintenance_price_vhigh" AND "maintenance_price_high" AND "maintenance_price_med" AND "maintenance_price_low" AND "doors_2" AND "doors_3" AND "doors_4" AND "doors_5more" AND "persons_2" AND "persons_4" AND "persons_more" AND "luggage_boot_size_small" AND "luggage_boot_size_med" AND "luggage_boot_size_big" AND "safety_low" AND "safety_med" AND "safety_high"'], [])
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
		0)


def test_synthetic_random_nulls():
	synth_test(
		test_id,
		'random',
		synth_patterns_cols,
		0)


def test_synthetic_80_percent_nulls():
	synth_test(
		test_id,
		'80-percent',
		synth_patterns_cols,
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

