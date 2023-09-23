import pandas as pd
import random

import sys
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker
from utils import synth_test, synth_test_all_cols, real_test, build_default_results

test_id = 'FEW_NEIGHBORS'
random.seed(0)

synth_patterns_cols = []
synth_exceptions_cols = ['few neighbors most', 'few neighbors date_most']


def test_real():
	res = build_default_results()
	res['isolet'] = ([], ['f408'])
	res['bioresponse'] = ([], 286)
	res['micro-mass'] = ([], 332)
	res['Amazon_employee_access'] = ([], ['RESOURCE'])
	res['abalone'] = ([], ['Height'])
	res['cnae-9'] = ([], ['V113', 'V303', 'V618', 'V715', 'V822'])
	res['vehicle'] = ([], ['SKEWNESS_ABOUT_MAJOR'])
	res['analcatdata_authorship'] = ([], ['are', 'be', 'even', 'will'])
	res['SpeedDating'] = ([], ['met'])
	res['credit-approval'] = ([], ['A11'])
	res['har'] = ([], 26)
	res['segment'] = ([], ['hedge-sd'])
	res['jm1'] = ([], 7)
	res['gas-drift'] = ([], 22)
	res['qsar-biodeg'] = ([], ['V7', 'V9', 'V11', 'V16', 'V32', 'V40', 'V41'])
	res['wdbc'] = ([], ['V4', 'V11', 'V13', 'V17', 'V20', 'V30'])
	res['ozone-level-8hr'] = ([], ['V72'])
	res['kc2'] = ([], ['ev(g)', 'n', 'v', 'd', 'e', 'b', 'lOBlank', 'uniq_Op', 'total_Op', 'total_Opnd'])
	res['eeg-eye-state'] = ([], ['V2', 'V3', 'V5', 'V8', 'V11', 'V12', 'V14'])
	res['spambase'] = ([], [
		'word_freq_address', 'word_freq_3d', 'word_freq_remove', 'word_freq_mail', 'word_freq_addresses',
		'word_freq_free', 'word_freq_email', 'word_freq_hpl', 'word_freq_650', 'word_freq_parts',
		'word_freq_pm', 'word_freq_cs', 'word_freq_original', 'word_freq_re', 'word_freq_conference',
		'char_freq_%3B', 'char_freq_%28', 'char_freq_%21', 'char_freq_%24', 'char_freq_%23'])
	res['ilpd'] = ([], ['V3', 'V7'])
	res['one-hundred-plants-margin'] = ([], ['V13', 'V16', 'V34', 'V61'])
	res['electricity'] = ([], ['vicprice'])
	res['scene'] = ([], ['attr151', 'attr161', 'attr190'])
	res['musk'] = ([], ['f76'])
	res['Click_prediction_small'] = ([], ['impression'])
	res['page-blocks'] = ([], ['height', 'blackpix'])
	res['hypothyroid'] = ([], ['TT4'])
	res['CreditCardSubset'] = ([], ['V6', 'V20'])
	res['shuttle'] = ([], ['A4', 'A8'])
	res['mc1'] = ([], [
		'BRANCH_COUNT', 'LOC_CODE_AND_COMMENT', 'EDGE_COUNT', 'HALSTEAD_ERROR_EST', 'HALSTEAD_LENGTH',
		'HALSTEAD_VOLUME', 'MODIFIED_CONDITION_COUNT', 'NODE_COUNT', 'NUM_OPERATORS'])
	res['pc1'] = ([], ['loc', 'v(g)', 'ev(g)', 'V', 'L', 'E', 'B', 'T', 'lOCode', 'locCodeAndComment', 'uniq_Opnd', 'branchCount'])
	res['wine-quality-white'] = ([], ['V1', 'V3'])
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


# def test_page_blocks():
# 	page_blocks_test(
# 		test_id,
# 		[],
# 		['height', 'blackpix']
# 	)
#
#
# def test_hypothyroid():
# 	hypothyroid_test(
# 		test_id,
# 		[],
# 		['TT4']
# 	)
