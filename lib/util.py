from typing import List, Union, Callable, Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import dask.dataframe as dd

import plotly.express as px
import plotly.graph_objs as go
import folium

def read_multiple_csv(
    fp:Union[Path, str],
    engine:str='pandas',
    **kwargs
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Read multiple CSV in an efficient RAM usage

    Parameters
    ----------
    fp : Union[Path, str]
        folderpath to your CSVs
    engine : str, optional
        dataframe engine, can be 'dask' or 'pandas', by default 'pandas'
    **kwargs:
        argument to pass to .read_csv()

    Returns
    -------
    Union[pd.DataFrame, dd.DataFrame]
        your DataFrame with all your CSV lodaded
    """
    
    if engine == 'pandas':
        eng = pd
    elif engine == 'dask':
        eng = dd
    else:
        NotImplementedError(f'{engine} is not valid')

    return eng.concat([eng.read_csv(x, **kwargs) for x in fp.rglob('*.csv')])


def plot_multilple_series(
    *series:List[pd.Series],
    kind:Optional[str]='line',
    **kwargs
):
    """
    plot multiple pandas time-series with plotly (interactive plot)

    Parameters
    ----------
    kind : str, optional
        kind of plot, can be 'line', 'scatter', or other, this will provide you with a dictionnary of the argument to the plotting function
        , by default 'line'
    *series:
        your series to concatenate and plot

    Returns
    -------
        a plotly.express figure, else a dictionnary
    """
    if kind == 'line':
        plot = px.line
    elif kind == 'scatter':
        plot = px.scatter
    else:
        plot = dict

    data = series[0].to_frame().join(series[1:], how='outer')

    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines',
            name=col,
            connectgaps=True
        ))
    
    return fig
 
def repartition_by(df:dd.DataFrame, by:Optional[str]=None) -> dd.DataFrame:
    """
    Repartition a Dask.DataFrame on a specified column (or index if none given).
    This accelerate processes by letting the user use "map_partition()" instead 
    of "groupby()".
    The "by" options is not as robust as desired.

    Parameters
    ----------
    df : dd.DataFrame
        Dask DataFrame to repartition
    by : Optional[str]
        column by which to repartition, None by default, (eq. index)

    Returns
    -------
    dd.DataFrame
        Repartitionned Dask DataFrame. You can now apply "map_partition()" 
        instead of "groupby()"
    """

    if by:
        index_name = df.index.name
        if index_name is None:
            index_name = 'index'
        try:
            df = df.reset_index()
        except ValueError:
            # in case the index is already a column
            df = df.reset_index(drop=True).set_index(by, drop=False)
        df = df.set_index(by, drop=False)
    
    # Create divisions based on index, need to be sorted
    # //HACK: Dask use the length of the divisions as  the number of 
    # partitions, hence one last added division (equal to last value)
    index_division = sorted(list(df.index.unique().compute()))
    index_division = [*index_division, index_division[-1]]

    df = df.repartition(divisions=index_division)

    if by:
        df = df.set_index(index_name)

    return df


def plot_gps(
    df:pd.DataFrame,
    lat:str='fix_latitude',
    lon:str='fix_longitude'
) -> folium.Map:

    gpsmap = folium.Map(
        location=df[[lat, lon]].median().to_list(),
        tiles='openstreetmap',
        zoom_start=15
    )

    # Add a circle for each row
    for idx, row in df.iterrows():
        folium.Circle(
            location=[row[lat], row[lon]],
            radius=10,
            color='forestgreen'
        ).add_to(gpsmap)

    return gpsmap


def most_frequent_for_category(x):
    res = x.mode()
    if len(res) == 0:
        return np.NaN
    else:
        return res[0]


def agg_num_and_obj(
    x:pd.Series,
    num_func:Optional[Callable]=np.mean,
    obj_func:Optional[Callable]=most_frequent_for_category
):

    if pd.api.types.is_numeric_dtype(x):
        return num_func(x)
    else:
        return obj_func(x)


def read_parquet_and_prepare(
    filepath:Path,
    col_repartition='trip',
    col_reindex='timestamp'
) -> pd.DataFrame:

    df = dd.read_parquet(filepath)

    # Depends if "timestamp" or something else is an index...
    if df.index.name is not None:
        try:
            df = df.reset_index(drop=False)
        except ValueError:
            # Means the column already exist, no need to retrieve it
            pass

    if col_repartition is not None:
        # I repartitions based on a column, to avoid "groupby", and instead 
        # use "map_partitions"
        df = repartition_by(df.set_index(col_repartition, drop=False))

        if col_reindex is None:
            df = df.assign(idx=1)\
                   .assign(idx=lambda df: df.idx.cumsum() - 1)
            df = df.map_partitions(lambda x: x.set_index('idx').sort_index())
        else:    
            # change the index to "timestamp" (make sense for a time-series) and
            # sort it (when loading it losts its order sometime)
            df = df.map_partitions(
                lambda x: x.set_index(col_reindex, drop=False).sort_index()
            )

    return df


def dask_agg_extent():
    return dd.Aggregation(
        name='extent',
        chunk=lambda grouped: (grouped.max(), grouped.min()),
        agg=lambda chunk_max, chunk_min: (chunk_max.max(), chunk_min.min()),
        finalize=lambda maxima, minima: maxima - minima
    )


def dask_agg_absmax():
    return dd.Aggregation(
        name='absmax',
        chunk=lambda grouped: abs(grouped.max()),
        agg=lambda chunk_max: abs(chunk_max.max())
    )


def dask_agg_largest():
    return dd.Aggregation(
        name='largest',
        chunk=lambda grouped: (grouped.max(), grouped.min()),
        agg=lambda chunk_max, chunk_min: (chunk_max.max(), chunk_min.min()),
        finalize=lambda M, m: np.sign(M + m) * abs(pd.concat([M, m], axis=1)).max(axis=1)
    )


def agg_multilple(
    gb:pd.core.groupby.generic.DataFrameGroupBy,
    setup:Dict[Tuple[str], List]
) -> pd.DataFrame:

    res = None
    for key, value in setup.items():
        res_temp = gb[list(key)].agg(value)
        if isinstance(value, list):
            res_temp = res_temp.rename(
                columns=lambda x: '_'.join(x) if isinstance(x, tuple) else x
            )
        
        if res is None:
            res = res_temp
        else:
            # Same index, just need to join
            res = res.join(res_temp)

    # Rename columns name from tuple to string with "_" separator
    res.columns = res.columns.map(
        lambda x: '_'.join(x) if isinstance(x, tuple) else x
    )
    return res


def reset_index_dask(ddf:dd.DataFrame) -> dd.DataFrame:
    return ddf.assign(idx=1)\
              .assign(idx=lambda df: df.idx.cumsum() - 1)\
              .set_index('idx', sorted=True)\
              .map_partitions(lambda df: df.rename(index = {'idx': None}))
