import pandas as pd
from pandas.io.formats.style import Styler

from .plotting_utils import get_names

def make_table(layerwise_counts):
    _, combo_name_dict, layerwise_counts = get_names(layerwise_counts)
    #print(combo_name_dict, layerwise_counts.keys())
    df = pd.DataFrame.from_dict(layerwise_counts, orient='columns')
    df.rename(
        columns=combo_name_dict,
        inplace=True,#not strictly necessary but for clarity
    )
    df.index.name = 'Layer'
    df.loc['Total'] = df.sum()
    styler = Styler(df)
    return styler
