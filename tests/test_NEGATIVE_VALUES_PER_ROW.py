import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker

from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'NEGATIVE_VALUES_PER_ROW'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = ['This test executes over all numeric columns']

def test_real():
	res = build_default_results()
	res['bioresponse']              = (1, 0)
	res['soybean']                  = (1, 0)
	res['micro-mass']               = (1, 0)
	res['Amazon_employee_access']   = (1, 0)
	res['abalone']                  = (1, 0)
	res['cnae-9']                   = (1, 0)
	res['semeion']                  = (1, 0)
	res['vehicle']                  = (1, 0)
	res['analcatdata_authorship']   = (1, 0)
	res['breast-w']                 = (1, 0)
	res['eucalyptus']               = (1, 0)
	res['wall-robot-navigation'] = (1, 0)
	res['credit-approval'] = (1, 0)
	res['splice'] = (1, 0)
	res['cmc'] = (1, 0)
	res['segment'] = (0, 1)
	res['jm1'] = (1, 0)
	res['mushroom'] = (1, 0)
	res['irish'] = (1, 0)
	res['profb'] = (1, 0)
	res['adult'] = (1, 0)
	res['anneal'] = (1, 0)
	res['credit-g'] = (1, 0)
	res['blood-transfusion-service-center'] = (1, 0)
	res['monks-problems-2'] = (1, 0)
	res['tic-tac-toe'] = (1, 0)
	res['wdbc'] = (1, 0)
	res['diabetes'] = (1, 0)
	res['hill-valley'] = (1, 0)
	res['kc2'] = (1, 0)
	res['eeg-eye-state'] = (1, 0)
	res['climate-model-simulation-crashes'] = (1, 0)
	res['spambase'] = (1, 0)
	res['ilpd'] = (1, 0)
	res['one-hundred-plants-margin'] = (1, 0)
	res['mozilla4'] = (1, 0)
	res['electricity'] = (1, 0)
	res['madelon'] = (1, 0)
	res['scene'] = (1, 0)
	res['nomao'] = (1, 0)
	res['Click_prediction_small'] = (1, 0)
	res['nursery'] = (1, 0)
	res['page-blocks'] = (1, 0)
	res['hypothyroid'] = (1, 0)
	res['yeast'] = (1, 0)
	res['kropt'] = (1, 0)
	res['shuttle']                      = (0, 1)
	res['Satellite']                    = (1, 0)
	res['baseball'] = (1, 0)
	res['mc1'] = (1, 0)
	res['pc1'] = (1, 0)
	res['kr-vs-k'] = (1, 0)
	res['volcanoes-a1'] = (1, 0)
	res['wine-quality-white'] = (1, 0)
	res['car-evaluation'] = (1, 0)
	res['solar-flare'] = (1, 0)
	res['allbp'] = (1, 0)
	res['allrep'] = (1, 0)
	res['dis'] = (1, 0)
	res['car'] = (1, 0)
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
		0,  # Given nulls, there are an inconsistent number of negative values from row to row.
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
