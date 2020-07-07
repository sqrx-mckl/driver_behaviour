import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Dict, Union, List, Callable, Optional
import folium
from folium.plugins import MarkerCluster

def plot_xy_max_and_min(df, x, y, groupby):
    df_plot = df.groupby(groupby)\
                [[x, y]]\
                .agg(['max', 'min'])\
                .stack()
    return sns.jointplot(x, y, df_plot, kind='hex', cmap='jet')


def absmax(x):
    return abs(max(x))
def extent(x):
    return x.max() - x.min()

def plot_gx_gy(df):
    plot_xy_max_and_min(
        df, x='gl2y', y='gl1x', groupby='corner_select'
    )

def plot_gy_speed(df):
    df = df.groupby('corner_select').agg({
        'gl2y':[absmax, extent],
        'sp1':'max'
    })
    df.columns = df.columns.to_flat_index()
    df = df.rename(columns=lambda x: '_'.join(x))
    
    _, axes = plt.subplots(1,2, figsize=(14, 8))
    for k, x in enumerate(['gl2y_absmax', 'gl2y_extent']):
        df.plot.hexbin(x=x, y='sp1_max', cmap='jet', gridsize=20, ax=axes[k])


def plot_line_events(df:pd.DataFrame, x:str, y:str, hue:str):
    return df.set_index(x)\
             .groupby(hue)[y]\
             .plot(style='k', alpha=0.02)


# --- D E E P   L E A R N I N G ---

from tensorflow_core.python.keras.models import Sequential, Input, Model
from tensorflow_core.python.keras.layers import (
    Bidirectional, Dropout, TimeDistributed,
    BatchNormalization, PReLU, ELU,
    Concatenate, RepeatVector, Subtract,
    LSTM, Dense
)


def demo_create_encoder(latent_dim, cat_dim, window_size, input_dim):
    input_layer = Input(shape=(window_size, input_dim))
    
    code = TimeDistributed(Dense(64, activation='linear'))(input_layer)
    code = Bidirectional(LSTM(128, return_sequences=True))(code)
    code = BatchNormalization()(code)
    code = ELU()(code)
    code = Bidirectional(LSTM(64))(code)
    code = BatchNormalization()(code)
    code = ELU()(code)
    
    cat = Dense(64)(code)
    cat = BatchNormalization()(cat)
    cat = PReLU()(cat)
    cat = Dense(cat_dim, activation='softmax')(cat)
    
    latent_repr = Dense(64)(code)
    latent_repr = BatchNormalization()(latent_repr)
    latent_repr = PReLU()(latent_repr)
    latent_repr = Dense(latent_dim, activation='linear')(latent_repr)
    
    decode = Concatenate()([latent_repr, cat])
    decode = RepeatVector(window_size)(decode)
    decode = Bidirectional(LSTM(64, return_sequences=True))(decode)
    decode = ELU()(decode)
    decode = Bidirectional(LSTM(128, return_sequences=True))(decode)
    decode = ELU()(decode)
    decode = TimeDistributed(Dense(64))(decode)
    decode = ELU()(decode)
    decode = TimeDistributed(Dense(input_dim, activation='linear'))(decode)
    
    error = Subtract()([input_layer, decode])
        
    return Model(input_layer, [decode, latent_repr, cat, error])


def demo_create_discriminator(latent_dim):
    input_layer = Input(shape=(latent_dim,))
    disc = Dense(128)(input_layer)
    disc = ELU()(disc)
    disc = Dense(64)(disc)
    disc = ELU()(disc)
    disc = Dense(1, activation="sigmoid")(disc)
    
    model = Model(input_layer, disc)
    return model


def demo_sample_normal(latent_dim, batch_size, window_size=None):
    shape = (batch_size, latent_dim) if window_size is None else (batch_size, window_size, latent_dim)
    return np.random.normal(size=shape)
  

def demo_sample_categories(cat_dim, batch_size):
    cats = np.zeros((batch_size, cat_dim))
    for i in range(batch_size):
        one = np.random.randint(0, cat_dim)
        cats[i][one] = 1
    return cats