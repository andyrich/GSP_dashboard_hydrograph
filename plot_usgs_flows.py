import plotly.express as px
import plotly.graph_objects as go
import helper
import pandas as pd
from plotly.subplots import make_subplots
import os
import numpy as np
import plotly.colors

def load():
    def dload(station):
        flow, info = helper.download_daily(station, begin_year=2018)

        flow.loc[:, 'Julian Date'] = helper.julian_water_year(flow.index)
        flow.loc[:, 'Water Year'] = helper.water_year(flow.index)

        return {'flow': flow, 'info': info}

    sites = pd.read_csv(os.path.join('assets', 'usgs_sites.csv'))


    lake_son_usgs = sites.loc[sites.loc[:, 'Lake Sonoma'] == 1]
    lake_men_usgs = sites.loc[sites.loc[:, 'Lake Mendocino'] == 1]
    lake_pills_usgs = sites.loc[sites.loc[:, 'Lake Pillsbury'] == 1]

    # load usgs flows
    lake_son_flows = {x: dload(x) for x in lake_son_usgs.site_no.unique()}
    lake_men_flows = {x: dload(x) for x in lake_men_usgs.site_no.unique()}
    lake_pills_flows = {x: dload(x) for x in lake_pills_usgs.site_no.unique()}

    return lake_son_flows, lake_men_flows, lake_pills_flows


def plot_timeseries_flows(x, title):
    print('dodo\n'*5)
    print(x.head())
    # Create a scatter plot for the timeseries flows
    trace = go.Scatter(
        x=x.reset_index().loc[:,"Date"],
        y=x["Q"],
        mode="lines",
        name=title,
        # line=dict(color=color),
        hovertemplate='%{y:,d} <i>af</i>',
    )

    layout = go.Layout(
        title=title,
        # log_y=True,
        hovermode="x",
        xaxis=dict(
            tickformat="%y %b",
            title="Date",
        ),
        yaxis=dict(
            title="cfs",
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 20},
    )
    return trace, layout

def plot_water_year_flows(x, title, colors = None, idx = 0):
    # Extract the maximum water year for trace customization
    yearmax = x["Water Year"].max()

    # Create a scatter plot for the water year flows
    traces = []
    for water_year in x["Water Year"].unique():
        # name = str(water_year) if idx == 0 else None
        color = colors[water_year]
        filtered_data = x[x["Water Year"] == water_year]
        traces.append(go.Scatter(
            x=filtered_data["Julian Date"],
            y=filtered_data["Q"],
            mode="lines",
            name=str(water_year),
            line = dict(color=color),
            showlegend= idx == 0,
            # line=dict(width=7 if water_year == yearmax else 2),  # Highlight the max year
            hovertemplate='%{y:,d} <i>af</i>',

        ))

    layout = go.Layout(
        title=title,
        # log_y=True,
        hovermode="x",
        xaxis=dict(
            tickformat="%b %d",
            title="Julian Date",
        ),
        yaxis=dict(
            title="cfs",
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 20},
    )
    return traces, layout




def plot_all(dict_of_stations, option):

    n = len(dict_of_stations.keys())
    rows = (n + 1) // 2  # Calculate the number of rows (2 columns per row)

    cur_year = pd.to_datetime('now').year+2
    # Example
    n = np.arange(2014, cur_year)
    colors = get_plotly_colors_discrete(len(n))
    colors = dict(zip(n,colors))

    if option=='yearly':
        # Create subplot grid
        fig = make_subplots(rows=rows, cols=2)
    else:
        fig = go.Figure()

    for idx, station in enumerate(dict_of_stations.keys()):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # for station in dict_of_stations.keys():

        if option == 'yearly':

            # Add water year flows plot
            water_year_traces, _ = plot_water_year_flows(dict_of_stations[station]['flow'],
                                                         dict_of_stations[station]['info']['Site Name'].values[0],
                                                         colors = colors, idx = idx)
            for trace in water_year_traces:
                fig.add_trace(trace, row=row, col=col)

            fig.update_yaxes(type="log", row=row, col=col)
            fig.update_xaxes(
                                tickformat="%b %d",
                                title="Julian Date",
                                 row = row, col = col
                             )

            # Customize subplot layout
            fig.update_layout(height=600 * rows )

        else:
            # Add timeseries flows plot
            timeseries_trace, _ = plot_timeseries_flows(dict_of_stations[station]['flow'],
                                                        dict_of_stations[station]['info']['Site Name'].values[0])

            fig.add_trace(timeseries_trace)
            fig.update_yaxes(type="log")
            fig.update_layout(height=600)




    return fig



def get_plotly_colors_discrete(n):
    palette = plotly.colors.DEFAULT_PLOTLY_COLORS
    return [palette[i % len(palette)] for i in range(n)]

