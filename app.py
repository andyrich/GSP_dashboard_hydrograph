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
import wiski_data
# dash.register_page(__name__, path='/')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

df_meas = pd.read_csv(os.path.join('obs_data_for_website.csv'))
df_meas.loc[:, 'Timestamp'] = pd.to_datetime(df_meas.loc[:, 'Timestamp'])

path = os.path.join('allinfo_stations.csv')

allinfo = pd.read_csv(path, index_col=[0])
allinfo = allinfo.rename(columns={'Station Name.1': 'Station Name'})
allinfo.loc[:, 'Station Name'] = allinfo.index


def get_loc(name):
    loci = allinfo[allinfo.loc[:, 'Station Name'] == name].loc[:, 'Latitude':'Longitude']

    return [[loci.at[name, 'Latitude']], [loci.at[name, 'Longitude']]]


all_options = {
    'Santa Rosa Plain': sorted(df_meas.query("Site=='SRP'").station_name.unique()),
    'Sonoma Valley': sorted(df_meas.query("Site=='SON'").station_name.unique()),
    'Petaluma Valley': sorted(df_meas.query("Site=='PET'").station_name.unique())
}


def layout():
    return html.Div([
        html.H1("Waterlevel Hydrographs"),
        html.Div(
            [html.Div([
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
                    value=["Shallow (0-200ft)", "Medium (200-500ft)",
                           "Deep(>500ft)"], ), ],
                style={'width': '20%', 'display': 'inline-block'}
            ),

                html.Div([
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
                    )],
                    style={'width': '20%', 'float': 'middle', 'display': 'inline-block'}
                ),

                html.Div([dcc.Graph(id='mapbox'), ],
                         style={'width': '50%', 'float': 'right', 'height': '500', 'display': 'inline-block'}),
            ]),

        html.Hr(),
        html.Button("Show Hydrograph", id="show-image", n_clicks=0),
        dcc.Graph(id='graph'),
        html.Hr(),
        html.H4("Station Info"),
        dcc.Markdown('''''', id='well_info'),
        # dcc.Graph(id = 'table', ),

        # dcc.Checklist(id = 'depth',
        #               options = ['Deep', 'Shallow', 'Unknown'],
        #               value = ['Deep', 'Shallow'],
        #                 ),
        # dcc.Dropdown(
        #     id='countries-radio',
        #     options=[{'label': k, 'value': k} for k in all_options.keys()],
        #     value='Santa Rosa Plain',
        #     placeholder="Select a Basin",
        # ),

        # dcc.Dropdown(id='cities-radio',
        #     placeholder="Select a station",),
    ])

app.layout = layout()

# @app.callback(
#     Output('cities-radio', 'options'),
#     Input('countries-radio', 'value'))
# def set_cities_options(selected_country):
#     return [{'label': i, 'value': i} for i in all_options[selected_country]]
#
#
# @app.callback(
#     Output('cities-radio', 'value'),
#     Input('cities-radio', 'options'))
# def set_cities_value(available_options):
#     return available_options[0]['value']

# @app.callback(
#     Output('cities-radio', 'value'),
#     Input('cities-radio', 'options'))
# def set_cities_value(available_options):
#     if available_options is None:
#         out = ''
#     else:
#         out = available_options['station_name']
#     return out

# #update table
@callback(
    Output('well_info', 'children'),
    Input('mapbox', 'selectedData'))
def set_table_value(available_options):
    cur = available_options
    if cur is None:
        return go.Figure(data=[go.Table()]), ''''''
    else:
        cur = cur['points'][0]['hovertext']
        print(cur)

    # table = go.Figure(data=[go.Table(
    # header=dict(values=list(allinfo.loc[:,:'APN Number'].columns),
    #             fill_color='paleturquoise',
    #             align='left'),
    # cells=dict(values=allinfo.loc[[cur],:'APN Number'].T.to_numpy().tolist(),
    #            fill_color='lavender',
    #            align='left'))])

    vals = allinfo.loc[[cur], :'APN Number'].T.dropna().to_dict()
    vals = vals[cur]
    d = [f"{k[0]}: {k[1]}" for k in vals.items()]
    mdown = '  \n'.join(d)
    print(mdown)

    return mdown


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
        raise PreventUpdate

    if colorscale is None:
        dfi = pd.DataFrame(columns=["Timestamp", "Manual Measurement", 'name'])
        colorscale = ""
    else:
        colorscale = colorscale['points'][0]['hovertext']
        print(colorscale)

        dfi = df_meas.query(f"station_name=='{colorscale}'")

    title = f"{colorscale}"

    x =wiski_data.wiski_plot(colorscale)

    x.get_station_pars(remove_pt=True)
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
     Input('checkbox', 'value')]
)
def update_figure(colorscale, depth):
    if colorscale is None:
        colorscale = 'Son0001'
    # def update_figure(colorscale):
    if 'all' in [x.lower() for x in depth]:
        cdf = allinfo
    else:
        cdf = allinfo.query(f"Well_Depth_Category=={depth}")

    fig = px.scatter_mapbox(cdf, lat="Latitude", lon="Longitude", hover_name="Station Name",
                            color='Well_Depth_Category',

                            hover_data=['Station Number', 'Local Grid ID', 'StateID',
                                        'State Well Completion Report Number', 'CASGEM', ])
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_geos(fitbounds="locations")
    fig.update_layout(clickmode="event+select")
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
