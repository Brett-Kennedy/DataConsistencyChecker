import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'MATCHED_MISSING'
random.seed(0)

synth_patterns_cols = [
	'"matched_missing_vals rand_a" AND "matched_missing_vals all"'
]
synth_exceptions_cols = [
	'"matched_missing_vals rand_a" AND "matched_missing_vals most"',
	'"matched_missing_vals all" AND "matched_missing_vals most"'
]


def test_real():
	res = build_default_results()
	res['soybean'] = ([
		'"precip" AND "stem-cankers" AND "canker-lesion" AND "external-decay" AND "mycelium" AND "int-discolor" AND "sclerotia"',
		'"hail" AND "severity" AND "seed-tmt" AND "lodging"',
		'"crop-hist" AND "plant-growth" AND "stem"',
		'"leafspots-halo" AND "leafspots-marg" AND "leafspot-size" AND "leaf-malf"',
		'"fruiting-bodies" AND "fruit-spots" AND "seed-discolor" AND "shriveling"',
		'"seed" AND "mold-growth" AND "seed-size"'
	], [])
	res['SpeedDating'] = (4, 256)
	res['eucalyptus'] = (['"Vig" AND "Ins_res" AND "Stem_Fm" AND "Crown_Fm" AND "Brnch_Fm"'], [])
	res['credit-approval'] = (['"A4" AND "A5"', '"A6" AND "A7"'], 4)
	res['adult'] = ([], ['"workclass" AND "occupation"'] )
	res['hypothyroid'] = ([], ['"T4U" AND "FTI"'])
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
		[],  # The nulls do not match when nulls are added randomly
		[])


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
		[],
		[])


def test_synthetic_all_cols_80_percent_nulls():
	synth_test_all_cols(
		test_id,
		'80-percent',
		synth_patterns_cols,
		synth_exceptions_cols)
