
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc


import pandas as pd


# dash.register_page(__name__)

# kurl = 'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?'
# k = KIWIS(kurl, verify_ssl=False)
import helper

k = helper.get_kiwis()

def get_men():
  wrs_cdec = k.get_timeseries_values(ts_id=2012010, **{'from': '2014-01-01'}, keep_tz=False)
  wrs_usace= k.get_timeseries_values(ts_id=50958010, **{'from': '2014-01-01'}, keep_tz=False)
  wrs_cdec.index = wrs_cdec.index.tz_convert('America/Los_Angeles').tz_localize(None)
  wrs_usace.index = wrs_usace.index.tz_convert('America/Los_Angeles').tz_localize(None)

  x = pd.concat([wrs_cdec, wrs_usace])
  x = x.resample('1D').mean().sort_index()
  x['year'] = x.index.year
  x['day'] = x.index.dayofyear

  stor= k.get_timeseries_values(ts_id=46520010, **{'from': '2014-01-01'}, keep_tz=False)
  stor.index = stor.index.tz_convert('America/Los_Angeles').tz_localize(None)
  stor['day'] = stor.index.dayofyear
  stor= stor.groupby('day').mean()
  # stor = stor.resample('1D').mean().sort_index()
  stor['dayofyear'] = stor.index


  return x, stor


def plot_men(x, stor):
    fig = px.line(x, x="day", y="Value", color='year', width=1200, height=600,
                  title="Lake Mendocino Storage",
                  labels={
                      "Timestamp": "Day of Year",
                      "Value": "acre-feet",

                  }, )

    fig.update_layout(hovermode="x")

    fig.add_trace(go.Scatter(x=stor.dayofyear, y=stor.Value, fill='tonexty',
                             mode='none', line_color='black',
                             fillcolor='rgba(135,206,235,.5)', name='Storage Curve'))
    fig.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
    fig.update_xaxes(tickformat="%b %d")
    return fig




def get_son():
  #lake sonoma
  wrs_cdec = k.get_timeseries_values(ts_id=2010010, **{'from': '2014-01-01'}, keep_tz=False)
  wrs_usace= k.get_timeseries_values(ts_id=50965010, **{'from': '2014-01-01'}, keep_tz=False)
  wrs_cdec.index = wrs_cdec.index.tz_convert('America/Los_Angeles').tz_localize(None)
  wrs_usace.index = wrs_usace.index.tz_convert('America/Los_Angeles').tz_localize(None)

  x = pd.concat([wrs_cdec, wrs_usace])
  x = x.resample('1D').mean().sort_index()
  x['year'] = x.index.year
  x['day'] = x.index.dayofyear
  stor= k.get_timeseries_values(ts_id=46515010, **{'from': '2014-01-01'}, keep_tz=False)
  stor.index = stor.index.tz_convert('America/Los_Angeles').tz_localize(None)
  stor['day'] = stor.index.dayofyear
  stor= stor.groupby('day').mean()
  stor['dayofyear'] = stor.index

  return x, stor

def plot_son(x, stor):
  fignew = px.line(x, x="day", y="Value", color='year',width=1200, height=600,
                title="Lake Sonoma Storage",
                labels={
                      "dy": "Day of Year",
                      "Value": "acre-feet",

                  },)



  fignew.update_layout(hovermode="x")

  fignew.add_trace(go.Scatter(x=stor.dayofyear, y=stor.Value, fill='tonexty',
                    mode= 'none', line_color='black',
                    fillcolor = 'skyblue', name = 'Storage Curve'))
  fignew.update_xaxes(tickformat="%b %d")
  fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
  return fignew
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