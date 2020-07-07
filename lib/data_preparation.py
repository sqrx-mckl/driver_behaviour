import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import Dict, Union, List, Callable, Tuple, Optional

from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter, butter, sosfiltfilt, find_peaks

import ruptures as rpt

import matplotlib.pyplot as plt
import plotly.express as px
from lib.util import plot_multilple_series


def signal_resample(
    ser_raw:pd.Series,
    resample_method:str='linear',
    resample_timestep:str='100ms'
) -> pd.Series:
    """
    Resample the data and fill the values following 3 methods:
    * linear will apply:
        * linear interpolation when 1 point is missing
        * Akima interpolation when more points are missing
        * forward/backward filling for the outside points
    * piecewise will apply forward filling (keep same value while moving forward), and then apply backward fill
    * nan will keep the NaNs
    """

    oidx = ser_raw.index
    nidx = pd.date_range(oidx.min(), oidx.max(), freq=resample_timestep)

    if resample_method is 'piecewise':
        return ser_raw.reindex(oidx.union(nidx))\
                      .interpolate('ffill')\
                      .reindex(nidx)\
                      .ffill()\
                      .bfill()

    ser_resample = ser_raw.reindex(oidx.union(nidx))\
                        .interpolate('index')\
                        .reindex(nidx)

    if resample_method is 'linear':
        return ser_resample\
            .interpolate(method='linear', limit=1, limit_area='inside')\
            .interpolate(method='akima', limit_area='inside')\
            .bfill()\
            .ffill()

    return ser_resample


def signal_preprocess_linear(
    ser:pd.Series,
    sos:np.ndarray=None,
    output_type:str='final'
) -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series]]:
    """
    Method to smooth your signal with Savitzky-Golay filter
    (see scipy.signal.savgol_filter)
    
    Parameters
    ----------
    """

    # First, resample at uniform sampling rate the data and interpolate
    ser_interp = signal_resample(ser)

    # Setup low-pass filter from EuroNCAP criteria
    sos = sos or butter(5, 4, 'low', output='sos', fs=10)
    
    # Filter the data using filtfilt (filter in bidirectional)
    ser_filt = pd.Series(
        sosfiltfilt(sos, ser_interp.to_list()),
        index=ser_interp.index
    )

    # Last, smooth the data with Savitsky-Golay
    ser_smooth = pd.Series(
        savgol_filter(
            ser_filt.to_list(),
            window_length=5,
            polyorder=2
        ),
        index=ser_interp.index
    )

    # return different output depending on what user asks for (useful for test)
    if output_type == 'all':
        return (ser_interp, ser_filt, ser_smooth)
    else:
        return ser_smooth


def signal_integrate(x:pd.Series, time:Optional[pd.Series]=None):
    if time is None:
        time = pd.to_numeric(x.index) / 10**9

    return pd.Series(
        cumtrapz(x.to_list(), x=time, initial=0),
        index=x.index
    )

def signal_derivate(x:pd.Series, time:Optional[pd.Series]=None):
    if time is None:
        time = pd.to_numeric(x.index) / 10**9

    return pd.Series(np.gradient(x, time), x.index)


def signal_delta_integrate(x:pd.Series, time:pd.Series):
    return signal_integrate(x, time).diff().fillna(0)


def add_vd_signal(
    df:pd.DataFrame,
    time:Optional[pd.Series]=None,
    method='delta'
) -> pd.DataFrame:

    if time is None:
        df['time'] = pd.to_numeric(df['timestamp']) // 10**3 / 10**6
        time = df['time']

    df['gx'] = df['gx'] * 9.81
    df['gy'] = df['gy'] * 9.81
    df['gz'] = df['gz'] * 9.81

    df['yr'] = np.gradient(df['yaw'], time)
    
    df['beta'] = df['gy'] - (np.deg2rad(df['yr']) * (df['speed']/3.6))
    df['vx'] = np.cos(np.deg2rad(df['beta'])) * (df['speed']/3.6)
    df['vy'] = np.sin(np.deg2rad(df['beta'])) * (df['speed']/3.6)
    df['curvature'] = (df['speed']/3.6) / np.deg2rad(df['yr'])
    
    if method == 'delta':
        integration = signal_delta_integrate
    elif method == 'integral':
        integration = signal_integrate
    else:
        NotImplementedError(f'unknown integration method: {method}')

    ya_rad = np.deg2rad(df['yaw'])

    df['distx'] = integration(
        np.cos(ya_rad) * df['vx'] - np.sin(ya_rad) * df['vy'],
        time
    )

    df['disty'] = integration(
        np.sin(ya_rad) * df['vx'] + np.cos(ya_rad) * df['vy'],
        time
    )

    df['distance'] = integration(df['speed']/3.6, time)

    return df


def adaptative_sampling(x:pd.Series, std_dev:float=None) -> np.ndarray:
    """
    Apply Bottom-up segmentation and "findpeaks" for robustness.
    This algorithm will retrieve the change-point location.

    Parameters
    ----------
    x : pd.Series
        initial serie to find the change-point
    std_dev : float, optional
        standard-deviation of your serie, can be local or global value
        , by default None - revert to local value

    Returns
    -------
    np.ndarray
        boolean array which locates the change-points
    """
    if std_dev is None:
        std_dev = x.std()

    if x.shape[0] == 0:
        return pd.Series(dtype='float64', name=x.name)

    # piecewise-segmentation with BottomUp algorithm
    X = list(range(len(x)))
    signal = np.column_stack((x.to_numpy().reshape(-1, 1), X))
    bottom_up = rpt.BottomUp(model='linear', jump=10)\
        .fit_predict(signal, pen=std_dev*np.log(len(x)))

    # add the peaks to be robust
    peaks_p = find_peaks(x, prominence=std_dev)[0]
    peaks_n = find_peaks(-x, prominence=std_dev)[0]
    
    # concatenate and sort all
    segments = sorted(list(set([*peaks_p, *peaks_n, *bottom_up])))

    # convert from position to boolean
    cond = np.zeros(x.shape, dtype=bool)
    # last value in "segment" is the length
    cond[segments[:-1]] = True
    # convert all non-selected values to NaN
    return cond


def apply_signal_preparation(
    df_raw:pd.DataFrame,
    min_row:int=5
) -> pd.DataFrame:
    """
    Preprocess raw dataframe to obtain evenly sampled data with the following 
    method for the value filling:
    * interpolate all linear data such as speed, GPS
    * smooth all dynamic/accelero/sensor data such as YAW, GX, GY
    * nearest all other data

    If there is less than "min_row" rows in a signal, it applies "piecewise" instead of linear (too small data size for interpolation)

    Parameters
    ----------
    df_raw : pd.DataFrame
        raw data to process, MUST contain a Datetime as index
    min_row : int, optional
        number of row to move to default method "nearest", by default 5
    
    Returns
    -------
    pd.DataFrame
        processed data
    """

    if df_raw.shape[0] < 1:
        return df_raw

    # Special case where there is not enough data
    def check_notna(col:List[str], min_row=min_row):
        return [x for x in col if df_raw[x].notna().sum() > min_row]

    # Apply different resampling filling for each signal
    col_smooth = check_notna(['gx', 'gy', 'yaw', 'roll', 'pitch'])
    col_interpolate = check_notna([
        'latitude',
        'longitude',
        'speed',
    ])
    col_piecewise = df_raw.columns.difference(
        [*col_interpolate, *col_smooth, 'timestamp']
    )

    setup = zip(
        [col_smooth, col_interpolate, col_piecewise],
        [signal_preprocess_linear,
         lambda x: signal_resample(x, 'linear'),
         lambda x: signal_resample(x, 'piecewise')]
    )

    df = df_raw.copy()

    index_is_valid_time = isinstance(df_raw.index, pd.DatetimeIndex)
    
    if not index_is_valid_time:
        if isinstance(df_raw.index, pd.MultiIndex):
            NotImplementedError("Don't know what to do in case of a MultiIndex")
        
        index_name = df_raw.index.name
        if index_name is None:
            index_name = 'index'
        
        try:
            df = df.reset_index()
        except ValueError:
            # in case the index is already a column
            df = df.reset_index(drop=True)

        df = df.set_index('timestamp')
    
    df = pd.concat(
            [df[col].apply(func) for col, func in setup]
            ,axis=1
        ).reindex(columns=df_raw.columns)
    
    df['timestamp'] = df.index
    
    if not index_is_valid_time:
        try:
            df = df.set_index(index_name)
        except:
            df = df.reset_index(drop=True)

    return df


def apply_data_preparation_fast(
    df_input:dd.DataFrame,
    inplace:bool=False
) -> pd.DataFrame:
    """
    Apply the following:
    * resampling and interpolation, linear and piecewise depending on columns
    * add Vehicle Dynamic signals
    segmentation can be added afterward with "calculate_change_point()"

    Parameters
    ----------
    df_input : dd.DataFrame
        initial dataframe
    inplace : bool, optional
        should this be apply to itself, by default False

    Returns
    -------
    pd.DataFrame
        dataframe with prepared data
    """
    if inplace:
        df = df_input
    else:
        df = df_input.copy()

    # First we apply signal transformation (resampling and interpolation)
    # this is thought to be done trip by trip
    df = df.set_index('timestamp')\
        .groupby('trip')\
        .apply(apply_signal_preparation)\
        .reset_index(drop=True)

    # We add the Vehicle Dynamics signals
    df = df.groupby('trip').apply(add_vd_signal)

    return df


def calculate_change_point(
    df:dd.DataFrame,
    groupby:str='trip',
    col:List[str]=['gx','gy']
) -> pd.Series:
    """
    Apply piecewise-segmentation to one or multiple columns and apply and 
    retrieve any change-points from all the columns

    Parameters
    ----------
    df : dd.DataFrame
        Initial DataFrame which we retrieve the signals to apply
    groupby : str, optional
        column by which to group-by before applying the adptative-sampling/
        change-point
        by default 'trip'
    col : List[str], optional
        List of from which to get the change-points
        by default ['gx','gy']

    Returns
    -------
    pd.Series
        series which contains any change-points from all columns
    """

    # We add the segmented signal
    if isinstance(df, dd.DataFrame):
        std_dev = {k:df[k].std().compute() for k in col}
    elif isinstance(df, pd.DataFrame):
        std_dev = {k:df[k].std() for k in col}


    return df.groupby(groupby)[col]\
             .transform(lambda x: adaptative_sampling(x, std_dev[x.name]))\
             .any(axis='columns')


def apply_data_preparation(
    df_input:dd.DataFrame,
    inplace:bool=False
) -> pd.DataFrame:
    """
    Apply the following:
    * resampling and interpolation, linear and piecewise depending on columns
    * add Vehicle Dynamic signals
    * add segmentation change-points for some signals
    A fast version without the segmentation done is available (much faster)

    Parameters
    ----------
    df_input : dd.DataFrame
        initial dataframe
    inplace : bool, optional
        should this be apply to itself, by default False

    Returns
    -------
    pd.DataFrame
        dataframe with prepared data
    """
    if inplace:
        df = df_input
    else:
        df = df_input.copy()

    # First we apply signal transformation (resampling and interpolation)
    # this is thought to be done trip by trip
    df = df.set_index('timestamp')\
        .groupby('trip')\
        .apply(apply_signal_preparation)\
        .reset_index(drop=True)

    # We add the Vehicle Dynamics signals
    df = df.groupby('trip').apply(add_vd_signal)

    # We add the segmented signal
    col_seg = ['gx','gy']
    std_dev = {k:df[k].std().compute() for k in col_seg}

    df['change_points'] = df.groupby('trip')[col_seg]\
        .transform(lambda x: adaptative_sampling(x, std_dev[x.name]))\
        .any(axis='columns')

    return df


def plot_linear(ser:pd.Series):

    (ser_interp,
     ser_filt,
     ser_smooth) = signal_preprocess_linear(ser, output_type='all')

    return plot_multilple_series(
        ser.rename('raw'),
        signal_resample(ser, resample_method='').rename('resample'),
        ser_interp.rename('interpolate'),
        ser_filt.rename('filter'),
        ser_smooth.rename('smooth'),
        title=f'{ser.name} through different processing stage'
    )


def plot_segmentation(ser:pd.Series, std_dev:float):

    cond = adaptative_sampling(ser, std_dev=std_dev)
    compression_rate = 1 - (sum(cond) / len(cond))
    ser_segmented = ser[cond]

    fig = plot_multilple_series(
        ser.rename('raw'),
        ser_segmented.rename('segmented'),
        kind='scatter',
        title=f'{ser.name} - compression rate: {compression_rate*100:.1f}%'
    )
    [g.update(mode='lines') for g in fig.data]
    fig.data[-1].update(mode='lines+markers')
    return fig


def plot_linear_and_segmentation(ser:pd.Series, std_dev:float):

    (ser_interp,
     ser_filt,
     ser_smooth) = signal_preprocess_linear(ser, output_type='all')

    cond = adaptative_sampling(ser_smooth, std_dev=std_dev)
    compression_rate = 1 - (sum(cond) / len(cond))
    ser_segmented = ser_smooth[cond]

    fig = plot_multilple_series(
        ser.rename('raw'),
        signal_resample(ser, resample_method='').rename('resample'),
        ser_interp.rename('interpolate'),
        ser_filt.rename('filter'),
        ser_smooth.rename('smooth'),
        ser_segmented.rename('segmented'),
        kind='scatter',
        title=f'{ser.name} - compression rate: {compression_rate*100:.1f}%'
    )
    [g.update(mode='lines') for g in fig.data]
    fig.data[-1].update(mode='lines+markers')
    return fig
