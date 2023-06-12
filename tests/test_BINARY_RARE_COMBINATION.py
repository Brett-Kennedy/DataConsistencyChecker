import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'BINARY_RARE_COMBINATION'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = [
	'"bin_rare_combo all_a" AND "bin_rare_combo all_b" AND "bin_rare_combo all_c"',
	'"bin_rare_combo all_a" AND "bin_rare_combo all_b" AND "bin_rare_combo all_d"',
	'"bin_rare_combo all_a" AND "bin_rare_combo all_b" AND "bin_rare_combo all_e"',
	'"bin_rare_combo all_a" AND "bin_rare_combo all_c" AND "bin_rare_combo all_d"',
	'"bin_rare_combo all_a" AND "bin_rare_combo all_c" AND "bin_rare_combo all_e"',
	'"bin_rare_combo all_b" AND "bin_rare_combo all_c" AND "bin_rare_combo all_d"',
	'"bin_rare_combo all_b" AND "bin_rare_combo all_c" AND "bin_rare_combo all_e"',
	'"bin_rare_combo all_b" AND "bin_rare_combo all_d" AND "bin_rare_combo all_e"',
	'"bin_rare_combo all_c" AND "bin_rare_combo all_d" AND "bin_rare_combo all_e"'
]


def test_real():
	res = build_default_results()
	res['isolet'] = ([], ['"f578" AND "f579" AND "f580"'])
	res['soybean'] = ([], ['"plant-stand" AND "hail" AND "fruiting-bodies"'])
	res['mushroom'] = ([], ['"bruises%3F" AND "gill-spacing" AND "stalk-shape"'])
	res['scene'] = ([], [
		'"Beach" AND "Sunset" AND "Field"',
		'"Beach" AND "FallFoliage" AND "Field"',
		'"Beach" AND "Field" AND "Mountain"',
		'"FallFoliage" AND "Field" AND "Mountain"'
	])
	res['PhishingWebsites'] = ([], 158)
	res['hypothyroid'] = ([], [
		'"sex" AND "on_thyroxine" AND "T4U_measured"',
		'"sex" AND "on_thyroxine" AND "FTI_measured"',
		'"sex" AND "T4U_measured" AND "FTI_measured"',
		'"on_thyroxine" AND "T4U_measured" AND "FTI_measured"',
		'"T3_measured" AND "T4U_measured" AND "FTI_measured"',
	])
	res['mc1'] = ([], ['"DESIGN_DENSITY" AND "GLOBAL_DATA_DENSITY" AND "MAINTENANCE_SEVERITY"'])
	res['allbp'] = ([], [
		'"on_thyroxine" AND "T4U_measured" AND "FTI_measured"',
		'"T3_measured" AND "T4U_measured" AND "FTI_measured"',
	])
	res['allrep'] = ([], [
		'"on_thyroxine" AND "T4U_measured" AND "FTI_measured"',
		'"T3_measured" AND "T4U_measured" AND "FTI_measured"'
	])
	res['dis'] = ([], [
		'"on_thyroxine" AND "T4U_measured" AND "FTI_measured"',
		'"T3_measured" AND "T4U_measured" AND "FTI_measured"',
	])
	res['steel-plates-fault'] = ([], [
		'"V12" AND "V13" AND "V30"',
		'"V12" AND "V30" AND "V33"',
		'"V13" AND "V30" AND "V33"',
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
