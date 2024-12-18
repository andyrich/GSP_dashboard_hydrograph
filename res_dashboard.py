import os

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import reservoir_storage
import plotly.express as px
import helper
import get_stor_change_table
import precip_fig
from plotly.subplots import make_subplots
import get_precip_wy
import plot_outflows
import plot_usgs_flows

# Initialize the Dash app
app = dash.Dash(__name__)

lake_son_flows, lake_men_flows, lake_pills_flows = plot_usgs_flows.load()

lake_son_, lake_son_stor = reservoir_storage.get_son()
lake_son_ = lake_son_.reset_index()
lake_son_stor = lake_son_stor.reset_index(drop=True)

act_mendo, stor_mendo = reservoir_storage.get_men()
act_mendo = act_mendo.reset_index()
stor_mendo = stor_mendo.reset_index()

#data to load for the precip
options = ['Venado (Near Lake Sonoma)',
           # 'Santa Rosa Airport',
           'Ukiah Airport',
           # 'Sonoma (General Vallejo)'
           ]


dfall = get_precip_wy.get_allstations(options=options)
today = pd.Timestamp.now().strftime('%A, %B %d %Y')

print('act_mendo')
print(act_mendo.head())
print('stor_mendo')
print(stor_mendo.head())
print('lake_son_')
print(lake_son_.head())
print('lake_son_stor')
print(lake_son_stor.head())

sites = pd.read_csv(os.path.join('assets', 'usgs_sites.csv'))
lake_son_usgs = sites.loc[sites.loc[:, 'Lake Sonoma'] == 1]
lake_men_usgs = sites.loc[sites.loc[:, 'Lake Mendocino'] == 1]
lake_pills_usgs = sites.loc[sites.loc[:, 'Lake Pillsbury'] == 1]



#get outflows
lake_son_outflows, lake_men_outflows = plot_outflows.load_data()

def get_reservoir_figure_yearly(res="Lake Sonoma"):
    if res.lower() == "lake sonoma":
        fig = reservoir_storage.plot_son(lake_son_, lake_son_stor)
    elif res.lower() == 'lake mendocino':

        fig = reservoir_storage.plot_men(act_mendo, stor_mendo)
    else:
        fig = go.Figure()

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 20},  # Remove margins for better view
    )

    return fig


def get_reservoir_figure_timeseries(res="Lake Sonoma"):
    if res.lower() == "lake sonoma":
        fig = reservoir_storage.plot_reservoir_yearly(lake_son_, title="Lake Sonoma")
    elif res.lower() == 'lake mendocino':
        fig = reservoir_storage.plot_reservoir_yearly(act_mendo, title="Lake Mendocino")
    else:
        fig = go.Figure()

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 20},  # Remove margins for better view
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
app.layout = html.Div([
    # Top row: Reservoir selection and plot type
    html.Div([
        # First column: Reservoir selection
        html.Div([
            html.H2('Reservoir Selection:'),
            html.Div([
                html.Button("Lake Sonoma", id="btn-lake-sonoma", n_clicks=0),
                html.Button("Lake Mendocino", id="btn-lake-mendocino", n_clicks=0),
                html.Button("Lake Pillsbury", id="btn-pillsbury", n_clicks=0),
            ], style={"display": "flex", "justify-content": "flex-start", "gap": "10px"})
        ], style={"flex": "1"}),

        # Second column: Plot type
        html.Div([
            html.H2('Plot type:'),
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
        ], style={"flex": "1"})
    ], style={"display": "flex", "gap": "20px", "align-items": "center"}),

    # First plot: Takes the full width of the row
    html.Div([
        dcc.Graph(
            id="placeholder-figure",
            figure=get_reservoir_figure_yearly(res="Lake Sonoma")
        )
    ], style={"width": "100%", "margin": "10px 0"}),



    # Third row: Table and site map
    html.Div([
        html.Div(
            dcc.Graph(id="table1", figure=stor_table(res='Lake Sonoma')),
            style={"flex": "0 0 30%"}  # Table takes 30% width
        ),
        html.Div(
            dcc.Graph(id="sitemap", figure=helper.map_sites(sites, res='Lake Sonoma')),
            style={"flex": "1"}  # Map takes the remaining 70% width
        ),
    ], style={"display": "flex", "gap": "10px"}),  # Flexbox layout with spacing

    # Third row: Table and site map
    html.Div([
        html.Div(
            dcc.Graph(id="outflows", figure=plot_outflows.plot(lake_son_outflows, title = 'Lake Sonoma Outflows', option= "yearly")),
            style={"flex": "1"}  # Map takes the remaining 70% width
        ),
    ], style={"display": "flex", "gap": "10px"}),  # Flexbox layout with spacing

    # Section for observed precipitation
    html.H2(f'Observed Precipitation'),
    dcc.Graph(id="precip", figure=precip_fig.update_precip(dfall=dfall, station="Venado (Near Lake Sonoma)")),

    # Dynamic streamflow figures
    html.H2(f'Observed Streamflow'),
    dcc.Graph(id="dynamic-figures", figure = go.Figure())
])



# Callbacks
@app.callback(
    Output("placeholder-figure", "figure"),
    Output("dynamic-figures", "figure"),
    Output("table1", "figure"),
    Output("sitemap", "figure"),
    Output('precip', 'figure'),
Output('outflows', 'figure'),
    Input("btn-lake-sonoma", "n_clicks"),
    Input("btn-lake-mendocino", "n_clicks"),
    Input("btn-pillsbury", "n_clicks"),
    Input("radio-options", "value"),
    # Input('precipdropdown', 'value')
)
def update_dashboard(n_clicks_sonoma, n_clicks_mendocino, n_clicks_berryessa, option):
    selected_lake = "Lake Sonoma"
    if n_clicks_mendocino > n_clicks_sonoma and n_clicks_mendocino > n_clicks_berryessa:
        selected_lake = "Lake Mendocino"
    elif n_clicks_berryessa > n_clicks_sonoma and n_clicks_berryessa > n_clicks_mendocino:
        selected_lake = "Lake Pillsbury"

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
        precip_station = "Venado (Near Lake Sonoma)"
        out = lake_son_outflows
    elif selected_lake == "Lake Mendocino":
        d = lake_men_flows
        precip_station = 'Ukiah Airport'
        out = lake_men_outflows
    else:
        d = lake_pills_flows
        precip_station = 'Ukiah Airport'
        out = lake_son_outflows

    table = stor_table(res=selected_lake)

    siteMap = helper.map_sites(sites, res=selected_lake)

    outFlows = plot_outflows.plot(out, title= f"{selected_lake} Outflows", option = option)

    subplots = plot_usgs_flows.plot_all(dict_of_stations=d,option = option)
    # subplots = [subplots]
    # subplots = go.Figure()
    precFig = precip_fig.update_precip(precip_station, dfall)

    print("done making all plots\n"*10)
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

    return placeholder_figure, subplots, table, siteMap, precFig, outFlows



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
