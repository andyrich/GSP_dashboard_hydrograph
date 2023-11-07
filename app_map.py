import os
from plotly.tools import mpl_to_plotly
import plotly.express as px
from dash.dependencies import Input, Output
from dash import dash_table as dt
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash import callback
import dash
import dash.dcc as dcc
import dash.html as html
# import dash_core_components as dcc
# import dash_html_components as html
import pandas as pd
import numpy as np
import wiski_data
import helper
# dash.register_page(__name__, path='/')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# df_meas = pd.read_csv(os.path.join('obs_data_for_website.csv'))
# df_meas.loc[:, 'Timestamp'] = pd.to_datetime(df_meas.loc[:, 'Timestamp'])

k = helper.get_kiwis()
# pars = k.get_parameter_list(parametertype_name = "Groundw*", return_fields = ['station_name'])
ts = pd.concat([k.get_timeseries_list(station_name = 'SRP*',
                parametertype_name = "Groundw*",
                return_fields = ['station_name', 'coverage']),

                k.get_timeseries_list(station_name = 'Son*',
                parametertype_name = "Groundw*",
                return_fields = ['station_name', 'coverage','stationparameter_name']),

                k.get_timeseries_list(station_name = 'PET*',
                parametertype_name = "Groundw*",
                return_fields = ['station_name', 'coverage']),

                k.get_timeseries_list(station_name = 'LRR*',
                parametertype_name = "Groundw*",
                return_fields = ['station_name', 'coverage'])])

print('-----------')
print(ts.station_name.apply(lambda x:x[0:3]).unique())
print('-----------')

# path = os.path.join('allinfo_stations.csv')

# allinfo = pd.read_csv(path, index_col=[0])
# allinfo = allinfo.rename(columns={'Station Name.1': 'Station Name'})
# allinfo.loc[:, 'Station Name'] = allinfo.index

allinfo = pd.concat([k.get_station_list(
    return_fields =[ 'station_name', 'station_latitude','station_longitude', 'site_no','custom_attributes'],
    parametertype_name = "Groundw*", site_no = 'SRP*'),
    k.get_station_list(
    return_fields=['station_name', 'station_latitude', 'station_longitude',  'site_no','custom_attributes'],
    parametertype_name="Groundw*", site_no='Son'),
    k.get_station_list(
    return_fields=['station_name', 'station_latitude', 'station_longitude', 'site_no', 'custom_attributes'],
    parametertype_name="Groundw*", site_no='PET*'),
    k.get_station_list(
    return_fields=['station_name', 'station_latitude', 'station_longitude', 'site_no', 'custom_attributes'],
    parametertype_name="Groundw*", site_no='LRR*')
]
)
allinfo = allinfo.rename(columns={'station_name': 'Station Name'})
allinfo.index = allinfo.loc[:,'Station Name']

print(allinfo.columns)
print(allinfo.head())

def get_loc(name):
    loci = allinfo[allinfo.loc[:, 'Station Name'] == name].loc[:, ['Station Name','station_latitude','station_longitude']]

    return [[loci.at[name, 'station_latitude']], [loci.at[name, 'station_longitude']]]


all_options = {
    'Santa Rosa Plain': sorted(allinfo.query("site_no=='SRP'").loc[:,'Station Name'].unique()),
    'Sonoma Valley': sorted(allinfo.query("site_no=='Son'").loc[:,'Station Name'].unique()),
    'Petaluma Valley': sorted(allinfo.query("site_no=='PET'").loc[:,'Station Name'].unique()),
    'Lower Russian River': sorted(allinfo.query("site_no=='LRR'").loc[:,'Station Name'].unique())
}


app.layout = html.Div([
    html.H1("Waterlevel Hydrographs"),
    html.Div([
        html.Div([
            html.H1("Well Depth"),
            dcc.Checklist(
                id="checkbox",
                options=[
                    {"label": "Shallow", "value": "Shallow (0-200ft)"},
                    {"label": "Medium", "value": "Medium (200-500ft)"},
                    {"label": "Deep", "value": "Deep(>500ft)"},
                    {"label": "Unknown", "value": "Other"},
                    {"label": "All", "value": "All"},
                ],
                labelStyle={"display": "block"},
                value=["Shallow (0-200ft)", "Medium (200-500ft)", "Deep(>500ft)"],
            ),
            html.H1("Well Type"),
            dcc.Checklist(
                id="check_rmp",
                options=[
                    {"label": "RMP", "value": "RMP"},
                    {"label": "Non-RMP", "value": "Non-RMP"},
                    {"label": "All", "value": "All"},
                ],
                labelStyle={"display": "block"},
                value=["All"]
            ),
        ], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='mapbox'),
        ], style={'width': '70%', 'display': 'inline-block'}),
    ]),
    html.Hr(),
    html.Div([  # Create a new div for the range slider
        dcc.RangeSlider(
            id='depth-slider',
            min=1950,
            max=2023,
            step=5,
            marks={i: f"{i} ft" for i in range(1950, 2025, 5)},
            value=[2010, 2015],
        ),
    ], style={'width': '90%', 'margin': '0 auto', 'text-align': 'center'}),
    html.Hr(),
    html.Button("Show Hydrograph", id="show-image", n_clicks=0),
    dcc.Graph(id='graph'),
    html.Hr(),
])




# # #update table
# @callback(
#     Output('well_info', 'children'),
#     Input('mapbox', 'selectedData'))
# def set_table_value(available_options):
#     cur = available_options
#     if cur is None:
#         return go.Figure(data=[go.Table()]), ''''''
#     else:
#         cur = cur['points'][0]['hovertext']
#         print(cur)
#
#     # table = go.Figure(data=[go.Table(
#     # header=dict(values=list(allinfo.loc[:,:'APN Number'].columns),
#     #             fill_color='paleturquoise',
#     #             align='left'),
#     # cells=dict(values=allinfo.loc[[cur],:'APN Number'].T.to_numpy().tolist(),
#     #            fill_color='lavender',
#     #            align='left'))])
#
#     vals = allinfo.loc[[cur], :].T.dropna().to_dict()
#     # vals = allinfo.loc[[cur], :'APN Number'].T.dropna().to_dict()
#     vals = vals[cur]
#     d = [f"{k[0]}: {k[1]}" for k in vals.items()]
#     mdown = '  \n'.join(d)
#     print(mdown)
#
#     return mdown


# # # #update table
# @app.callback(
#     Output('well_info', 'children'),
#     Input('mapbox', 'selectedData'))
# def set_markdown_value(available_options):
#     if available_options is None:
#         return go.Figure(data=[go.Table()])
#     else:
#         print('--asdf')
#         print(available_options)
#         available_options = available_options['points'][0]['hovertext']
#         print(available_options)
#
#     wellinfo = '''asdfasdfasdf'''
#
#     return wellinfo


@callback(
    # Output('display-selected-values', 'children'),
    Output('graph', 'figure'),
    # Input('countries-radio', 'value'),
    Input('mapbox', 'selectedData'),
    Input("show-image", "n_clicks"),
    prevent_initial_call=True,
)
def update_figure(colorscale, n_clicks):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] != "show-image":
        print('preventing update')
        raise PreventUpdate

    if colorscale is None:
        dfi = pd.DataFrame(columns=["Timestamp", "Manual Measurement", 'name'])
        colorscale = "Son0001"
    else:
        colorscale = colorscale['points'][0]['hovertext']
        print(colorscale)

        # dfi = df_meas.query(f"station_name=='{colorscale}'")
    #
    title = f"{colorscale}"
    print(f"the name of the station is {title}")
    x =wiski_data.wiski_plot(colorscale)

    x.get_station_pars(remove_pt=False)
    fig = x.plot_gw()
    # plotly_fig = mpl_to_plotly(fig)

    # fig = px.scatter(
    #     dfi,
    #     x="Timestamp", y="Manual Measurement",
    #     color='name',
    #     render_mode="webgl", title=title
    # )
    return fig


@callback(
    Output('mapbox', 'figure'),
    [Input('mapbox', 'selectedData'),
     Input('checkbox', 'value'),
     Input('depth-slider', 'value')]  # Add this input
)
def update_figure(colorscale, depth, slider_value):  # Modify the function parameters
    if colorscale is None:
        colorscale = 'Son0001'

    # Filter the data based on the depth range selected with the slider
    yearmin, yearmax = slider_value

    if 'all' in [x.lower() for x in depth]:
        cdf = allinfo
    else:
        cdf = allinfo.query(f"Web_GW_Obs_Range=={depth}")

    ts.loc[:,'yearmin'] = pd.to_datetime(ts.loc[:,'from']).dt.year
    ts.loc[:, 'yearmax'] = pd.to_datetime(ts.loc[:, 'to']).dt.year

    # Filter the data based on the year range
    ts_file = ts[(ts['yearmin'] <= yearmin) & (ts['yearmax'] >= yearmax)]

    cdf = cdf.loc[cdf.loc[:,'Station Name'].isin(ts_file.loc[:,'station_name'])]
    def convert_empty_strings_to_nan(data_frame, column_name):
        # Replace empty strings with NaN in the specified column
        data_frame[column_name] = data_frame[column_name].replace('', np.nan)
        return data_frame

    cdf = convert_empty_strings_to_nan(cdf, 'station_longitude')
    cdf = convert_empty_strings_to_nan(cdf, 'station_latitude')

    cdf = cdf.astype({'station_longitude':np.float64,
                      'station_latitude': np.float64},  errors='ignore')
    cdf = cdf.dropna(subset = 'station_longitude')

    try:
        fig = px.scatter_mapbox(cdf,
                                lat="station_latitude",
                            lon="station_longitude",
                            hover_name="Station Name",
                            color='Web_GW_Obs_Range',

                            # hover_data=['Station Number', 'Local Grid ID', 'StateID',
                            #             'State Well Completion Report Number', 'CASGEM', ]

                            )




        fig.update_layout(mapbox_style="open-street-map")
        # fig.update_geos(fitbounds="locations")
        fig.update_layout(clickmode="event+select")

        print('the map figure does work')
    except:
        print('\n\nit DOES  fail\n\n')
        fig =  go.Figure()
    #
    # fig.add_trace(go.Scattermapbox(
    #     lat=get_loc(colorscale)[0],
    #     lon=get_loc(colorscale)[1],
    #     mode='markers+text',
    #     marker=go.scattermapbox.Marker(
    #         size=20,
    #         color='rgb(0, 0, 0)',
    #         opacity=1.00,
    #         symbol = 'circle'
    #     ),
    #
    #     name = colorscale,
    #     # hoverinfo='none',
    #     text = [colorscale]
    #
    # )
    # )

    # fig.update_geos(center = {'lat':get_loc(colorscale)[0][0], 'lon':get_loc(colorscale)[1][0] }, projection_scale = 1)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 8056)
