from pandas import DataFrame

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