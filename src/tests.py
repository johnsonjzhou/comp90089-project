################################################################################
# Test scripts                                                                 #
################################################################################
import logging

import pandas as pd
from pandas import DataFrame

def assert_group_identical(group:DataFrame) -> DataFrame:
    """
    Asserts that all rows are identical by checking columns only contain
    no more than one unique value
    
    Args:
        group (DataFrame)
    
    Returns(DataFrame)
    
    Throws(AssertionError)
    """
    # Group identifier
    g = group[["subject_id", "hadm_id"]].values.tolist()
    
    # Test for number of unique values in each column
    for col in group:
        if col == "stay_id":
            continue
        u = group[col].unique()
        assert len(u) == 1, \
            f"Group: {g}, col {col} has >= 1 unique values, being {u}"
    
    return

def validate_dataset(df:DataFrame):
    """
    Tests the validity of the dataset on the basis of:
    1. Each group of subject_id, hadm_id can only have one row
    2. There should not be any NA values
    3. The number of unique values in each column should be 1

    Logs an ERROR if any of the assertions do not pass with a message.
    Logs an INFO with success message if all assertions pass.
    
    Returns(bool): True if all tests pass
    """
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(format="%(levelname)s-%(message)s")
    
    def group_assertions(group:DataFrame) -> DataFrame:
        # Group identifier
        g = group[["subject_id", "hadm_id"]].values.tolist()
        
        # Test for NA values
        na_cols = group[group.columns[group.isna().any()]].columns.values
        assert len(na_cols) == 0, f"Group: {g}, NA present in columns: {na_cols}"
        
        # Test for number of unique values in each column
        for col in group:
            if col == "stay_id":
                continue
            u = group[col].unique()
            assert len(u) == 1, \
                f"Group: {g}, col {col} has >= 1 unique values, being {u}"
        
        # Test for number of records per group
        assert len(group) == 1, f"Group: {g}, length is {len(group)}"
        
        return
    
    try:
        # Test group assertions on the dataframe
        df.groupby(["subject_id", "hadm_id"]).apply(group_assertions)
        
        # If all assertions passed, print success message
        logging.info("Dataset is valid")
    except AssertionError as msg:
        logging.error(msg)
        return False

    return True

def test():
    df = pd.read_csv("../data/initial_cohort_final.csv", index_col=0)
    validate_dataset(df)
    return

if __name__ == "__main__":
    # test()
    pass