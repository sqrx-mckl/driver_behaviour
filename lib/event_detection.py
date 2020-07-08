import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Dict, Union, List, Callable, Optional
import folium
from folium.plugins import MarkerCluster


def count(flag:pd.Series) -> pd.Series:

    s = flag.astype(int)
    event = s.diff().fillna(0)

    # //HACK: cumsum the events starts, then reset the no-event rows to 0
    return (event > 0).cumsum() * s


def calculate(
    inside_flag:pd.Series,
    outside:pd.Series,
    outside_cond:Callable[[pd.Series], bool]
) -> pd.Series:

    not_event = count(~inside_flag)

    outside_flag = outside.groupby(not_event)\
                          .transform(lambda x: outside_cond(x))

    return count(inside_flag | outside_flag)


def make_hash(
    event_count:pd.Series,
    hash_prefix:Optional[pd.Series]=None,
    hash_suffix:Optional[pd.Series]=None,
    method:str='custom'
) -> pd.Series:

    # Retrieve only the first row of each event of the Series used for the hash
    df_hash = pd.concat([hash_prefix, event_count.rename('event'), hash_suffix],
                        axis=1)\
                .loc[event_count.astype(bool)]\
                .groupby(event_count)\
                .head(1)\
                .set_index('event', drop=False)
    #//TODO: maybe the set_index and head() can be replaced with transform()

    if method == 'custom':
        # add a separator between the date and the count
        df_hash['sep'] = 'x'
        df_hash = df_hash.iloc[:, [0,1,3,2]]

    # Concatenate all values per row into a single string
    hash_dict = df_hash.astype(str).sum(axis=1)
    # Replace the event_count by the hash
    s_hash = event_count.replace(hash_dict.to_dict())
    
    # replace the "0" category with np.nan
    s_hash = s_hash.replace({0:np.nan})

    if method != 'custom':
        # If not custom, we use Pandas function to calculate an integer hash
        s_hash = pd.util.hash_pandas_object(s_hash, index=False)
    
    return s_hash.astype('category')


def detect(
    outside_cond:Callable[[pd.Series], bool],
    inside_flag:Union[pd.Series, str],
    outside:Union[pd.Series, str],
    trip_id:Union[pd.Series, str]='trip',
    time:Union[pd.Series, str]='time',
    data:Optional[pd.DataFrame]=None
) -> pd.DataFrame:

    if data is not None:
        def replace_by_series(x:Union[pd.Series, str]) -> pd.Series:
            if isinstance(x, str):
                return data[x]
            else:
                return x

        # replace the "str" in pd.Series        
        inside_flag = replace_by_series(inside_flag)
        outside = replace_by_series(outside)
        trip_id = replace_by_series(trip_id)
        time = replace_by_series(time)

    return make_hash(
        calculate(inside_flag, outside, outside_cond),
        trip_id, time.astype(str).map(lambda x: x[-2:])
    )


def delete_event(
    event:pd.Series,
    event_bool=None,
    index=None,
    event_index=None,
    replace_by:Optional=np.NaN
) -> pd.Series:

    s = event.copy()

    if index is not None:
        s.iloc[index] = replace_by
    elif event_index is not None:
        s.loc[s.isin(event_index)] = replace_by
    elif event_bool is not None:
        event_index = event_bool[event_bool].index
        s.loc[s.isin(event_index)] = replace_by
    else:
        raise AttributeError('"cond" or "event_bool" need to be filled')
    
    return s.cat.remove_unused_categories()

def plot_gps_marked(
    df:pd.DataFrame,
    event:str,
    lat:str='latitude',
    lon:str='longitude',
) -> folium.Map:

    gpsmap = folium.Map(
        location=df[[lat, lon]].median().to_list(),
        tiles='openstreetmap',
        zoom_start=14
    )

    # Add a circle for each row
    for idx, row in df.iterrows():
        folium.Circle(
            location=[row[lat], row[lon]],
            radius=10,
            color='forestgreen'
        ).add_to(gpsmap)

    # if boolean interpretation of "event" is false it is a "no-event"
    gb = df.loc[df[event].astype(bool)].groupby(event)

    # add a marker for each event and new circles for each row in the event
    mc = MarkerCluster()
    for name, group in gb.__iter__():
        if group.empty:
            continue

        for idx, row in group.iterrows():
            folium.Circle(
                location=[row[lat], row[lon]],
                radius=10,
                color='darkred',
                fill=True,
                fill_color='red',
                fill_opacity=0.5
            ).add_to(gpsmap)

        mc.add_child(folium.Marker(group[[lat, lon]].median()))
    
    # Add the marker cluster which indicate "this is an event" to the plot
    gpsmap.add_child(mc)

    return gpsmap


def apply_corner_and_straight_event(
    df_input:pd.DataFrame
) -> pd.DataFrame:

    df = df_input.copy()

    df['dtime'] = df['time'].diff().fillna(0)

    df['corner_flag'] = (df['gy'].abs() > 0.3)
    df['straight_flag'] = (df['gy'].abs() < 0.5) & (df['speed'] > 10)

    corner_event = detect(
        data=df,
        inside_flag='corner_flag',
        outside='dtime',
        outside_cond=lambda x: x.sum() < 1
    )
    df['corner_event'] = delete_event(
        corner_event,
        event_bool=df['dtime'].groupby(corner_event).sum() < 3
    )

    straight_event = detect(
        data=df,
        inside_flag='straight_flag',
        outside='dtime',
        outside_cond=lambda x: x.sum() < 1
    )
    df['straight_event'] = delete_event(
        straight_event,
        event_bool=df['dtime'].groupby(straight_event).sum() < 3
    )
    
    return df


def change_points_event(df, event_col, select_col='change_points'):

    def select_points(df):
        if df.empty:
            return
        ix = df[select_col]
        ix.iloc[[0, -1]] = True
        return df.loc[ix]

    return df.groupby(event_col).apply(select_points)


def normalize_index(df, col='dtime'):
    return df.assign(
        idx=lambda df: df[col].cumsum() - df[col].iloc[0],
        idx_norm=lambda df: df['idx'] / df['idx'].iloc[-1]
    )