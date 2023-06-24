import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'DECISION_TREE_CLASSIFIER'
random.seed(0)

synth_patterns_cols = 1
	# '"dt cls. 1a" AND "dt cls. 1b" AND "dt cls. 2"'
synth_exceptions_cols = 1
	# '"dt cls. 1a" AND "dt cls. 1b" AND "dt cls. 3""'


# With these, different trees can legitamately be build, so we just track the number
# of columns where trees may be built. Some specific rules are used here if they are
# based on a single other column.
def test_real():
	res = build_default_results()
	res['isolet'] = ([], 2)
	# '"f584" AND "f579" AND "f325" AND "f578"'
	# '"f584" AND "f580" AND "f578" AND "f579"'

	res['bioresponse'] = (2, 0)
	# '"D29" AND "D28"'
	# '"D50" AND "D94"']

	res['soybean'] = (1, 0)
	# '"int-discolor" AND "external-decay" AND "sclerotia"'], [])

	res['SpeedDating'] = (9, 10)

	res['credit-approval'] = (3, 0)
	# '"A12" AND "A5" AND "A5" AND "A4"',
	# '"A4" AND "A4" AND "A1" AND "A5"',
	# '"A1" AND "A10"'

	res['anneal'] = (0, 2)
	# '"carbon" AND "hardness" AND "temper_rolling"',
	# '"surface-quality" AND "len" AND "hardness" AND "shape"'

	res['nomao'] = (2, 0)
	# '"V1" AND "V7"'
	# '"V2" AND "V8"' or '"V3" AND "V8"'

	res['hypothyroid'] = (1, 2)
	# '"T4U" AND "FTI" AND "FTI_measured"'
	# '"TT4" AND "T4U" AND "TT4_measured"',
	# '"T4U" AND "FTI" AND "T4U_measured"'

	res['mc1'] = (1, 0)
	# '"BRANCH_COUNT" AND "CYCLOMATIC_COMPLEXITY" AND "MAINTENANCE_SEVERITY"'

	res['car-evaluation'] = (21, 0)

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

	res['steel-plates-fault'] = (2, 0)
	# "V13" AND "V12"
	# "V12" AND "V13"

	real_test(test_id, res)


def test_synthetic_no_nulls():
	synth_test(
		test_id,
		'none',
		synth_patterns_cols,
		synth_exceptions_cols
	)


def test_synthetic_one_row_nulls():
	synth_test(
		test_id,
		'one-row',
		0,
		2)  # The pattern is now an exception


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
