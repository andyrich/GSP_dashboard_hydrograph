import plotly.express as px
import plotly.graph_objects as go
import helper
import pandas as pd
import os

def load():
    def dload(station):
        flow, info = helper.download_daily(station, begin_year=2018)

        flow.loc[:, 'Julian Date'] = helper.julian_water_year(flow.index)
        flow.loc[:, 'Water Year'] = helper.water_year(flow.index)

        return {'flow': flow, 'info': info}

    sites = pd.read_csv(os.path.join('assets', 'usgs_sites.csv'))
    print(sites.dtypes)

    lake_son_usgs = sites.loc[sites.loc[:, 'Lake Sonoma'] == 1]
    lake_men_usgs = sites.loc[sites.loc[:, 'Lake Mendocino'] == 1]
    lake_pills_usgs = sites.loc[sites.loc[:, 'Lake Pillsbury'] == 1]

    # load usgs flows
    lake_son_flows = {x: dload(x) for x in lake_son_usgs.site_no.unique()}
    lake_men_flows = {x: dload(x) for x in lake_men_usgs.site_no.unique()}
    lake_pills_flows = {x: dload(x) for x in lake_pills_usgs.site_no.unique()}

    return lake_son_flows, lake_men_flows, lake_pills_flows

def plot_timeseries_flows(x, info):

    fignew = px.line(x.reset_index(), x="Date", y="Q",
                     # color='Water Year',
                     # width=1200, height=600,
                     title=info.at[0, 'Site Name'],
                     log_y=True,
                     labels={
                         "dy": "Day of Year",
                         "Value": "acre-feet",

                     }, )

    fignew.update_layout(hovermode="x")
    fignew.update_xaxes(tickformat="%y %b")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
    fignew.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 20},  # Remove margins for better view
    )
    return fignew


def plot_water_year_flows(x, info):
    fignew = px.line(x, x="Julian Date", y="Q", color='Water Year',
                     # width=1200, height=600,
                     title=info.at[0, 'Site Name'],
                     log_y=True,
                     labels={
                         "dy": "Day of Year",
                         "Value": "acre-feet",

                     }, )

    fignew.update_layout(hovermode="x")

    fignew.update_xaxes(tickformat="%b %d")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
    fignew.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 20},  # Remove margins for better view
    )

    yearmax = x.loc[:,'Water Year'].max()

    fignew.for_each_trace(
        lambda trace: trace.line.update( width=7) if trace.name == f"{yearmax}" else (),
    )

    return fignew
