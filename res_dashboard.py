import os

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import reservoir_storage
import plotly.express as px
import helper
import get_stor_change_table
from plotly.subplots import make_subplots

# Initialize the Dash app
app = dash.Dash(__name__)

lake_son_, lake_son_stor = reservoir_storage.get_son()
lake_son_ = lake_son_.reset_index()
lake_son_stor = lake_son_stor.reset_index(drop=True)

act_mendo, stor_mendo = reservoir_storage.get_men()
act_mendo = act_mendo.reset_index()
stor_mendo = stor_mendo.reset_index()

print('act_mendo')
print(act_mendo.head())
print('stor_mendo')
print(stor_mendo.head())
print('lake_son_')
print(lake_son_.head())
print('lake_son_stor')
print(lake_son_stor.head())

sites = pd.read_csv(os.path.join('assets', 'usgs_sites.csv'))
print(sites.dtypes)

lake_son_usgs = sites.loc[sites.loc[:, 'Lake Sonoma'] == 1].head(2)
lake_men_usgs = sites.loc[sites.loc[:, 'Lake Mendocino'] == 1].head(2)
lake_pills_usgs = sites.loc[sites.loc[:, 'Lake Pillsbury'] == 1].head(2)


def dload(station):
    flow, info = helper.download_daily(station, begin_year=2018)

    flow.loc[:, 'Julian Date'] = helper.julian_water_year(flow.index)
    flow.loc[:, 'Water Year'] = helper.water_year(flow.index)

    return {'flow': flow, 'info': info}


lake_son_flows = {x: dload(x) for x in lake_son_usgs.site_no.unique()}
lake_men_flows = {x: dload(x) for x in lake_men_usgs.site_no.unique()}
lake_pills_flows = {x: dload(x) for x in lake_pills_usgs.site_no.unique()}


def get_reservoir_figure_yearly(res="Lake Sonoma"):
    if res.lower() == "lake sonoma":
        fig = reservoir_storage.plot_son(lake_son_, lake_son_stor)
    elif res.lower() == 'lake mendocino':

        fig = reservoir_storage.plot_men(act_mendo, stor_mendo)
    else:
        fig = go.Figure()

    return fig


def get_reservoir_figure_timeseries(res="Lake Sonoma"):
    if res.lower() == "lake sonoma":
        fig = reservoir_storage.plot_reservoir_yearly(lake_son_, title="Lake Sonoma")
    elif res.lower() == 'lake mendocino':
        fig = reservoir_storage.plot_reservoir_yearly(act_mendo, title="Lake Mendocino")
    else:
        fig = go.Figure()

    return fig


def plot_timeseries_flows(x, info):
    print(x.head().reset_index())
    fignew = px.line(x.reset_index(), x="Date", y="Q",
                     # color='Water Year',
                     width=1200, height=600,
                     title=info.at[0, 'Site Name'],
                     log_y=True,
                     labels={
                         "dy": "Day of Year",
                         "Value": "acre-feet",

                     }, )

    fignew.update_layout(hovermode="x")
    fignew.update_xaxes(tickformat="%y %b")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')

    return fignew


def plot_water_year_flows(x, info):
    fignew = px.line(x, x="Julian Date", y="Q", color='Water Year', width=1200, height=600,
                     title=info.at[0, 'Site Name'],
                     log_y=True,
                     labels={
                         "dy": "Day of Year",
                         "Value": "acre-feet",

                     }, )

    fignew.update_layout(hovermode="x")

    fignew.update_xaxes(tickformat="%b %d")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')

    return fignew


def map_sites(df, res='Lake Sonoma'):

    location = {"Lake Sonoma":(38.717799, -123.010034),
     "Lake Mendocino": (39.202599, -123.175258),
     "Lake Pillsbury": (39.408070, -122.954985)}

    cdf = df.loc[df.loc[:, res] == 1]

    assert cdf.shape[0]>0, f'shape of sites is {cdf}'

    cdf.loc[:,'size'] = 5

    yes = {x: True if x in ['site_no', 'station_nm', 'site_tp_cd', 'huc_cd', 'begin_date','end_date'] else False for x in cdf.columns}

    fig = px.scatter_mapbox(cdf,
                            lat="dec_lat_va",
                            lon="dec_long_va",
                            size='size',
                            hover_name="site_no",
                            hover_data=yes
                            )

    # Add the "like" symbol (e.g., heart emoji) and name
    fig.add_trace(go.Scattermapbox(
        lat=[location[res][0]],
        lon=[location[res][1]],
        mode='markers+text',
        marker=go.scattermapbox.Marker(size=14, symbol='circle', color='blue'),
        text=[res],
        textposition="top right",
        name=res
    ))

        # Add mapbox style and enable scroll zoom
    fig.update_layout(
        mapbox_style="open-street-map",  # Example map style, you can change it
        mapbox_zoom=8,  # Adjust initial zoom level
        # clickmode="event+select",
        mapbox_center={"lat": cdf['dec_lat_va'].mean(), "lon": cdf['dec_long_va'].mean()},
    )

    fig.update_layout(
        dragmode="pan",  # Enables panning
        margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Remove margins for better view
    )



    return fig


def stor_table(res="Lake Sonoma"):
    print("Getting storage table\n" * 5)
    if res.lower() == "lake sonoma":
        table_data = get_stor_change_table.get_storchange_table_for_reservoir(lake_son_, lake_son_stor, res)
    elif res.lower() == "lake mendocino":
        table_data = get_stor_change_table.get_storchange_table_for_reservoir(act_mendo, stor_mendo, res)
    else:
        return html.Div("No data available", style={"color": "red"})

    return table_data


# App layout
app.layout = html.Div([
    # First row: Buttons
    html.Div([
        html.Button("Lake Sonoma", id="btn-lake-sonoma", n_clicks=0),
        html.Button("Lake Mendocino", id="btn-lake-mendocino", n_clicks=0),
        html.Button("Lake Pillsbury", id="btn-pillsbury", n_clicks=0),
    ], style={"display": "flex", "justify-content": "space-around"}),

    # Second row: Radio options
    html.Div([
        dcc.RadioItems(
            id="radio-options",
            options=[
                {"label": "Yearly", "value": "yearly"},
                {"label": "Time Series", "value": "time_series"},
                {"label": "Peaks", "value": "peaks"}
            ],
            value="yearly",
            inline=True
        )
    ], style={"margin": "10px 0"}),

    # Placeholder figure and table
    dcc.Graph(id="placeholder-figure", figure=get_reservoir_figure_yearly(res="Lake Sonoma")),
    dcc.Graph(id="table1", figure=stor_table(res='Lake Sonoma')),
    dcc.Graph(id="sitemap", figure=map_sites(sites, res='Lake Sonoma')),

    # Dynamic figures
    html.Div(id="dynamic-figures")
])


# Callbacks
@app.callback(
    Output("placeholder-figure", "figure"),
    Output("dynamic-figures", "children"),
    Output("table1", "figure"),
    Output("sitemap", "figure"),
    Input("btn-lake-sonoma", "n_clicks"),
    Input("btn-lake-mendocino", "n_clicks"),
    Input("btn-pillsbury", "n_clicks"),
    Input("radio-options", "value")
)
def update_dashboard(n_clicks_sonoma, n_clicks_mendocino, n_clicks_berryessa, option):
    selected_lake = "Lake Sonoma"
    if n_clicks_mendocino > n_clicks_sonoma and n_clicks_mendocino > n_clicks_berryessa:
        selected_lake = "Lake Mendocino"
    elif n_clicks_berryessa > n_clicks_sonoma and n_clicks_berryessa > n_clicks_mendocino:
        selected_lake = "Lake Berryessa"

    if option == "yearly":
        # Placeholder figures and table
        placeholder_figure = get_reservoir_figure_yearly(res=selected_lake)
    elif option == "time_series":
        # Placeholder figures and table
        placeholder_figure = get_reservoir_figure_timeseries(res=selected_lake)
    else:
        placeholder_figure = go.Figure(title="Not Done")

    if selected_lake == "Lake Sonoma":
        d = lake_son_flows
    elif selected_lake == "Lake Mendocino":
        d = lake_men_flows
    else:
        d = lake_pills_flows

    table = stor_table(res=selected_lake)

    siteMap = map_sites(sites, res=selected_lake)

    # Placeholder logic for dynamic figures
    figures = []
    for station in d.keys():

        if option == 'yearly':
            fig = plot_water_year_flows(d[station]['flow'], d[station]['info'])
        else:
            fig = plot_timeseries_flows(d[station]['flow'], d[station]['info'])

        # figures.append([fig])
        figures.append(dcc.Graph(figure=fig))

    # n = len(figures)
    # rows = (n + 1) // 2  # Calculate the number of rows (2 columns per row)
    #
    # # Create subplot grid
    # fig = make_subplots(rows=rows, cols=2)
    #
    # for idx, plot in enumerate(figures):
    #     row = idx // 2 + 1
    #     col = idx % 2 + 1
    #     for p in plot:
    #         for trace in p.data:
    #             print(trace['x'])
    #             print(trace['y'])
    #             fig.add_trace((go.Scatter(x=trace['x'], y=trace['y'], name = trace['name'])), row=row, col=col)


    #

    return placeholder_figure, figures, table, siteMap


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
