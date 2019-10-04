import numpy as np
import pandas as pd

def logret(df, cols=None, to_cols=None, *, dropna=True, only_new=True):
    """ compute log returns given price
    cols: list of int or str
    to_cols: list of str for output, default col_logret
    """
    cols = df.columns if cols is None else cols
    to_cols = [col+'_logret' for col in cols] if to_cols is None else to_cols
    for col, to_col in zip(cols, to_cols):
        df[to_col] = np.log(df[col]) - np.log(df[col].shift(1))

    df = df if not dropna else df.dropna()
    df = df.drop(columns=list(set(cols)-set(to_cols))) if only_new else df
    return df