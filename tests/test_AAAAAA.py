"""
This file is executed first and will save a pickle of each dc object, allowing the
other tests to execute faster.
"""

# Standard imports
import dill
import os
import sys

# Our code
from list_real_files import real_files
from utils import TEST_REAL, load_openml_file
sys.path.insert(1, '..')
from check_data_consistency import DataConsistencyChecker


def test_init_cache():
    if not TEST_REAL:
        return

    # If not present, create the folder for the cache
    cache_folder = "dc_cache"
    os.makedirs(cache_folder, exist_ok=True)

    for dataset_name in real_files:
        file_name = os.path.join(cache_folder, dataset_name + "_dc.pkl")
        if os.path.exists(file_name):
            print(f"{dataset_name} already in cache")
        else:
            print(f"Initializing and caching {dataset_name}")
            data_df = load_openml_file(dataset_name)
            dc = DataConsistencyChecker(verbose=-1)
            dc.init_data(data_df)
            filehandler = open(file_name, 'wb')
            dill.dump(dc, filehandler)
