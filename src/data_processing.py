from pandas import DataFrame
import numpy as np

def remove_outliers(df:DataFrame) -> DataFrame:
    """
    Removes outliers being < (Q1-1.5IQR) or > (Q3+1.5IQR)
    Credit: https://stackoverflow.com/a/59366409/13522010
    
    Args:
        df (DataFrame)
    
    Returns (DataFrame)
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    df_trunc = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df_trunc

def any_one(group, startswith:str):
    """
    For columns beginning with 'startswith' parameter, if any row values
    has one or True, then cast entire column to one or True
    
    Args:
        group (DataFrame): groupby dataframe
        startswith (str): the start of the column name
    
    Returns(DataFrame)
    """
    # Base case
    if len(group) == 1:
        return group
    
    # Get target columns
    target_cols = [col for col in group.columns.values.tolist()
                 if col.startswith(startswith)]
    
    # If any row in column contains a 1 or True, the column will be 1 or True
    for col in target_cols:
        group[col] = np.where(np.any(group[col]), 1, 0)
    
    return group