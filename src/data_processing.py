from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

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

def forward_back_fill(group:DataFrame):
    """
    Fill NA with two strategies:
    1. Forward fill first, then
    2. Back fill
    
    Args:
        group (DataFrame): groupby dataframe
    
    Returns(DataFrame)
    """
    # Base case
    if len(group) == 1:
        return group
    
    # Firstly apply forward fill
    group = group.fillna(method="ffill")
    # Then, back fill
    group = group.fillna(method="bfill")
    
    return group

def case_aggregation(group:DataFrame, cases:list=None):
    """
    Apply a chosen aggregate function to a dataframe or group on a case
    by case basis
    
    Args:
        group (DataFrame): groupby dataframe
        cases (list): list of column suffix(str) and applicable function as a tuple
    
    Returns(DataFrame)
    """
    # Base case
    if len(group) == 1:
        return group
    
    if cases is None:
        # Default pairs of column suffix and applicable function
        cases = [
            ("_min", np.min),
            ("_max", np.max)
        ]
    
    for suffix, func in cases:
        # Get all columns ending with the suffix
        cols = group.columns.str.endswith(suffix)
        # Apply the func to the respective columns
        group.loc[:, cols] = \
            group.loc[:, cols].apply(func, result_type="broadcast")
    
    return group

def convert_inconsistent_uom(df):
    """
    Applies a unit conversion to named columns when values fall within
    a particular range
    
    Args:
        df(DataFrame)
    
    Returns(DataFrame)
    """
    df_converted = df.copy()
    for col in df.columns:
        # Default, no change
        convert_func = lambda x: x
        
        # Temperature, convert from celcius to fahrenheit
        if col.startswith("temperature"):
            convert_func = lambda x: np.where(
                (x <= 53), ((x * 9 / 5) + 32), x
            )
        
        # Height, convert from inches to cm
        if col.startswith("height"):
            convert_func = lambda x: np.where(
                (x <= 72), (x * 2.54), x
            )
        
        df_converted[col] = df_converted[col].apply(convert_func)
    
    return df_converted

def plot_cohort_statistics(df, title:str, figsize=(10,10), across:int=4):
    """
    Use a boxplot to visualise the numerical features
    
    Args:
        df (DataFrame)
        title (str): Title for the plot
        figsize (tuple): Size of the figure to pass to pyplot.figure
        across (int): How many subplot columns to display
    """
    _, col_count = df.shape
    col = across
    row = math.ceil(col_count/across)
    width, height = figsize
    fig, axs = plt.subplots(row, col)
    fig.set_size_inches(width, height)
    fig.suptitle(title, fontweight="semibold", y=1.0)
    fig.tight_layout(w_pad=2)
    
    df_plt = df.copy()

    for i, column in enumerate(df_plt):
        r = math.floor(i / col)
        c = i % col
        try:
            sns.boxplot(y=df_plt[column], ax=axs[r][c])
        except:
            print(f"Column {column} is not numeric")
            continue
    
    plt.show()
    return

def plot_df_histogram(df, title:str, figsize:tuple=(10,4), across:int = 4, **kwargs):
    """
    Plots a histogram from the dataframe
    
    Args:
        df (DataFrame)
        title (str): Title for the plot
        figsize (tuple): Size of the figure to pass to pyplot.figure
        across (int): How many subplot columns to display
    """
    _, col_count = df.shape
    df.hist(figsize=figsize, layout=(math.ceil(col_count/across), across), **kwargs)
    plt.suptitle(title, fontweight="semibold", y=1.0)
    plt.tight_layout()
    plt.show()
    return

def calculate_feature_correlation(df:DataFrame):
    """
    Compare features and calculate pearson correlation, return the absolute
    pearson coefficient and the signed direction
    
    Args:
        df(DataFrame)
    
    Returns(DataFrame)
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    # Wrangle into easy to read format
    # We want absolute pearson coefficient and direction [-1, 1]
    df_corr = DataFrame(corr_matrix.unstack(), columns=["pearson"])
    df_corr.index.names = ["X1", "X2"]
    df_corr = df_corr.reset_index().dropna()
    df_corr["pearson_abs"] = df_corr["pearson"].abs()
    df_corr["pearson_dir"] = df_corr["pearson_abs"] / df_corr["pearson"]
    df_corr = df_corr.drop(columns=["pearson"])
    # Ignore self correlations and sort descending
    df_corr = df_corr.loc[(df_corr["X1"] != df_corr["X2"])] \
        .sort_values(by="pearson_abs", ascending=False)
    return df_corr