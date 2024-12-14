
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc
import numpy as np
from datetime import datetime
import pandas as pd


# dash.register_page(__name__)

# kurl = 'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?'
# k = KIWIS(kurl, verify_ssl=False)
import helper

k = helper.get_kiwis()

def get_men():
    wrs_cdec = k.get_timeseries_values(ts_id=2012010,  **{'from': '2013-10-01'}, keep_tz=False)
    wrs_usace= k.get_timeseries_values(ts_id=50958010, **{'from': '2013-10-01'}, keep_tz=False)
    wrs_cdec.index = wrs_cdec.index.tz_convert('America/Los_Angeles').tz_localize(None)
    wrs_usace.index = wrs_usace.index.tz_convert('America/Los_Angeles').tz_localize(None)

    x = pd.concat([wrs_cdec, wrs_usace])
    x = x.resample('1D').mean().sort_index()
    x['Water Year'] = water_year(x.index)
    x['day'] = x.index.dayofyear
    x.loc[:, 'Date'] = days_into_water_year(x.index)

    stor= k.get_timeseries_values(ts_id=46520010, **{'from': '2018-01-01'}, keep_tz=False)
    stor.index = stor.index.tz_convert('America/Los_Angeles').tz_localize(None)

    stor.loc[:, 'new_days'] = stor.index
    stor.loc[stor.index.month >= 10, 'new_days'] = stor.loc[stor.index.month >= 10, :].index - pd.to_timedelta(365,
                                                                                                               unit='D')
    stor = stor.set_index('new_days')

    # stor['day'] = stor.index.dayofyear
    # stor= stor.groupby('day').mean()
    stor = stor.resample('1D').mean().sort_index()
    # stor['dayofyear'] = stor.index

    stor.loc[:, 'Date'] = days_into_water_year(stor.index)



    return x, stor


def plot_men(x, stor):
    fig = px.line(x, x="Date", y="Value", color='Water Year', width=1200, height=600,
                  title="Lake Mendocino Storage",
                  labels={
                      "Timestamp": "Day of Year",
                      "Value": "acre-feet",

                  }, )

    fig.update_layout(hovermode="x")

    fig.add_trace(go.Scatter(x=stor.Date, y=stor.Value, fill='tonexty',
                             mode='none', line_color='black',
                             fillcolor='rgba(135,206,235,.5)', name='Storage Curve'))
    fig.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
    fig.update_xaxes(tickformat="%b %d")
    yearmax = x.loc[:,'Water Year'].max()
    print(yearmax)
    fig.for_each_trace(
        lambda trace: trace.line.update( width=7) if trace.name == f"{yearmax}" else (),
    )

    print(x.head())
    return fig




def get_son():
    #lake sonoma
    wrs_cdec = k.get_timeseries_values(ts_id=2010010, **{'from': '2013-10-01'}, keep_tz=False)
    wrs_usace= k.get_timeseries_values(ts_id=50965010, **{'from': '2013-10-01'}, keep_tz=False)
    wrs_cdec.index = wrs_cdec.index.tz_convert('America/Los_Angeles').tz_localize(None)
    wrs_usace.index = wrs_usace.index.tz_convert('America/Los_Angeles').tz_localize(None)

    x = pd.concat([wrs_cdec, wrs_usace])
    x = x.resample('1D').mean().sort_index()
    x['Water Year'] = water_year(x.index)
    x['day'] = x.index.dayofyear
    x.loc[:,'Date'] = days_into_water_year(x.index)

    stor= k.get_timeseries_values(ts_id=46515010, **{'from': '2013-10-01'}, keep_tz=False)
    stor.index = stor.index.tz_convert('America/Los_Angeles').tz_localize(None)
    stor.loc[:, 'new_days'] = stor.index
    stor.loc[stor.index.month >= 10, 'new_days'] = stor.loc[stor.index.month >= 10, :].index - pd.to_timedelta(365,
                                                                                                               unit='D')
    stor.index = stor.new_days
    # stor= stor.groupby('day').mean()
    # stor['dayofyear'] = stor.index
    stor = stor.resample("1D").mean()

    stor.loc[:, 'Date'] = days_into_water_year(stor.index)


    return x, stor

def plot_son(x, stor):
    fignew = px.line(x, x="Date", y="Value", color='Water Year',width=1200, height=600,
                title="Lake Sonoma Storage",
                labels={
                      "dy": "Day of Year",
                      "Value": "acre-feet",

                  },)



    fignew.update_layout(hovermode="x")

    fignew.add_trace(go.Scatter(x=stor.Date, y=stor.Value, fill='tonexty',
                    mode= 'none', line_color='black',
                    fillcolor = 'skyblue', name = 'Storage Curve'))
    fignew.update_xaxes(tickformat="%b %d")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
    #set line width to thicker for current year
    yearmax = x.loc[:,'Water Year'].max()
    fignew.for_each_trace(
      lambda trace: trace.line.update(width=7) if trace.name == f"{yearmax}" else (),
    )

    return fignew

def plot_reservoir_yearly(storage_ts, title = "Lake Sonoma Storage"):
    fignew = px.line(storage_ts, x="Timestamp",
                     y="Value",
                     width=1200, height=600,
                title=title,
                labels={
                      "dy": "Day of Year",
                      "Value": "acre-feet",

                  },)

    fignew.update_layout(hovermode="x")
    fignew.update_xaxes(tickformat="%y %b")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')

    return fignew

def to_WY(date):

    """
    Calculate the number of days since October 1st for the given date.

    Parameters:
    - input_date (datetime): The date for which to calculate the days since October 1st.

    Returns:
    - int: The number of days since October 1st.
    """
    if date.month<10:
        # Get October 1st of the same year as input_date
        october_first = datetime(date.year-1, 10, 1)
    else:
        # Get October 1st of the same year as input_date
        october_first = datetime(date.year, 10, 1)

    # Calculate the difference in days
    delta = date - october_first

    new_date = datetime(2018,10,1) + delta

    return new_date


def days_into_water_year(date):
    '''
	this returns an integer water year of the date
	'''

    if isinstance(date, pd.Series):
        return date.apply(to_WY)
    if isinstance(date, datetime):
        return to_WY(date)
    elif isinstance(date, pd.DatetimeIndex):
        return [to_WY(i) for i in date]
    else:
        import warnings
        warnings.warn('not a Series/datetime/DatetimeIndex object')
        # print('not a Series/datetime/DatetimeIndex object')
        return np.nan

def water_year(date):
    '''
	this returns an integer water year of the date
	'''

    def wy(date):
        if date.month < 10:
            return date.year
        else:
            return date.year + 1

    if isinstance(date, pd.Series):
        return date.apply(wy)
    if isinstance(date, datetime):
        return wy(date)
    elif isinstance(date, pd.DatetimeIndex):
        return [wy(i) for i in date]
    else:
        import warnings
        warnings.warn('not a Series/datetime/DatetimeIndex object')
        # print('not a Series/datetime/DatetimeIndex object')
        return np.nan

#
# act_mendo, stor_mendo = get_men()
# act_mendo = act_mendo.reset_index()
# stor_mendo = stor_mendo.reset_index()
# fig = plot_men(act_mendo, stor_mendo)
#
#
# x, stor = get_son()
# fig2 = plot_son(x.reset_index(), stor)

# def layout():
#     return html.Div(children=[
#         html.H1(children='Reservoir Storage'),
#
#         dcc.Graph(
#             id='example-graph',
#             figure=fig
#         ),
#
#
#         dcc.Graph(
#             id='ex2',
#             figure=fig2
#         )
#
#     ])