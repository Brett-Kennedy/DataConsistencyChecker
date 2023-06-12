import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'BINARY_IMPLIES'
random.seed(0)

synth_patterns_cols = [
	'"bin implies all_1" AND "bin implies all_2"'
]
synth_exceptions_cols = [
	'"bin implies all_1" AND "bin implies most"',
	'"bin implies all_2" AND "bin implies most"'
]


def test_real():
	res = build_default_results()
	res['isolet'] = ([
		'"f578" AND "f579"',
		'"f578" AND "f580"',
		'"f579" AND "f580"'], [])
	res['semeion'] = (0, 16)
	res['scene'] = ([
		'"Beach" AND "Sunset"',
		'"Beach" AND "FallFoliage"',
		'"Sunset" AND "FallFoliage"',
		'"Sunset" AND "Field"',
		'"Sunset" AND "Mountain"'
		], [
		'"Beach" AND "Field"'
	])
	res['nomao'] = (['"V7" AND "V8"'], [])
	res['PhishingWebsites'] = ([], [
		'"Favicon" AND "port"',
		'"on_mouseover" AND "popUpWidnow"'])
	res['hypothyroid'] = (0, ['"T4U_measured" AND "FTI_measured"'])
	res['cardiotocography'] = ([
		'"V26" AND "V27"',
		'"V26" AND "V31"',
		'"V26" AND "V32"',
		'"V27" AND "V31"',
		'"V27" AND "V32"',
		'"V31" AND "V32"'
		], [])
	res['car-evaluation'] = ([
		'"buying_price_vhigh" AND "buying_price_high"',
		'"buying_price_vhigh" AND "buying_price_med"',
		'"buying_price_vhigh" AND "buying_price_low"',
		'"buying_price_high" AND "buying_price_med"',
		'"buying_price_high" AND "buying_price_low"',
		'"buying_price_med" AND "buying_price_low"',
		'"maintenance_price_vhigh" AND "maintenance_price_high"',
		'"maintenance_price_vhigh" AND "maintenance_price_med"',
		'"maintenance_price_vhigh" AND "maintenance_price_low"',
		'"maintenance_price_high" AND "maintenance_price_med"',
		'"maintenance_price_high" AND "maintenance_price_low"',
		'"maintenance_price_med" AND "maintenance_price_low"',
		'"doors_2" AND "doors_3"',
		'"doors_2" AND "doors_4"',
		'"doors_2" AND "doors_5more"',
		'"doors_3" AND "doors_4"',
		'"doors_3" AND "doors_5more"',
		'"doors_4" AND "doors_5more"',
		'"persons_2" AND "persons_4"',
		'"persons_2" AND "persons_more"',
		'"persons_4" AND "persons_more"',
		'"luggage_boot_size_small" AND "luggage_boot_size_med"',
		'"luggage_boot_size_small" AND "luggage_boot_size_big"',
		'"luggage_boot_size_med" AND "luggage_boot_size_big"',
		'"safety_low" AND "safety_med"',
		'"safety_low" AND "safety_high"',
		'"safety_med" AND "safety_high"'
	], [])
	res['allbp'] = (0, ['"T4U_measured" AND "FTI_measured"'])
	res['allrep'] = (0, ['"T4U_measured" AND "FTI_measured"'])
	res['dis'] = (0, ['"T4U_measured" AND "FTI_measured"'])
	res['steel-plates-fault'] = (
		[
			'"V12" AND "V13"',
			'"V30" AND "V33"'
		], [
			'"V12" AND "V30"',
			'"V13" AND "V30"'
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
		0,
		3)  # The pattern is now an exception


def test_synthetic_in_sync_nulls():
	synth_test(
		test_id,
		'in-sync',
		0,  # We should consider making these more robust to nulls.
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

