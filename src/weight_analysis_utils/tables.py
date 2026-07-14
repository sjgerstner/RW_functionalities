import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from .plotting_utils import get_names

def layerwise_count_df(layerwise_counts):
    _, combo_name_dict, layerwise_counts = get_names(layerwise_counts)
    #print(combo_name_dict, layerwise_counts.keys())
    df = pd.DataFrame.from_dict(layerwise_counts, orient='columns')#keys of dict become columns
    df.rename(
        columns=combo_name_dict,
        inplace=True,#not strictly necessary but for clarity
    )
    df.index.name = 'Layer'
    df.loc['Total'] = df.sum()
    return df

def quartile_df(
    list_data:list[np.ndarray], subtitles:list[str],
):
    quantiles = [np.quantile(data, q=[.25, .5, .75]) for data in list_data]
    df = pd.DataFrame.from_dict(
        {
            "name": subtitles,
            "mean": [data.mean() for data in list_data],
            "quartile1": [quantile[0] for quantile in quantiles],
            "median": [quantile[1] for quantile in quantiles],
            "quartile3": [quantile[2] for quantile in quantiles],
        },
        orient='columns',
    )
    return df

def df_to_nice_latex(df:pd.DataFrame, path, **kwargs):
    styler = Styler(df)
    styler.to_latex(path, **kwargs)
