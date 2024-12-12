import datetime
import os
from plotly.tools import mpl_to_plotly
import plotly.express as px
from dash.dependencies import Input, Output
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

import wiski_data_plot_multi
import wiski_data
import helper

# dash.register_page(__name__, path='/')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# Create the range of months and labels for slider
months = list(range(1, 19))
labels = [str(m) if m in [1, 2, 4, 6, 12, 18] else '' for m in months]
# df_meas = pd.read_csv(os.path.join('obs_data_for_website.csv'))
# df_meas.loc[:, 'Timestamp'] = pd.to_datetime(df_meas.loc[:, 'Timestamp'])

k = helper.get_kiwis()
# pars = k.get_parameter_list(parametertype_name = "Groundw*", return_fields = ['station_name'])
print('getting timeseries')


def get_ts():
    if os.path.exists('ts.pickle'):
        print('loading ts pickle')
        ts = pd.read_pickle('ts.pickle')
    else:
        print('loading ts from wiski')
        ts = pd.concat([k.get_timeseries_list(station_name='SRP*',
                                              parametertype_name="Groundw*",
                                              return_fields=['station_name', 'coverage', 'stationparameter_name']),

                        k.get_timeseries_list(station_name='Son*',
                                              parametertype_name="Groundw*",
                                              return_fields=['station_name', 'coverage', 'stationparameter_name']),

                        k.get_timeseries_list(station_name='PET*',
                                              parametertype_name="Groundw*",
                                              return_fields=['station_name', 'coverage', 'stationparameter_name']),

                        k.get_timeseries_list(station_name='LRR*',
                                              parametertype_name="Groundw*",
                                              return_fields=['station_name', 'coverage', 'stationparameter_name'])])

        ts.to_pickle('ts.pickle')
    return ts


def get_gw_mon_status(station):
    '''
    get the gw_monitiring station dataa:
        'OwnerMon, MonSiteFreq, MonSGMASiteCode, MonSGMA, MonRMP, MonCASGEM, MonAgency, LastPressMeas, LastManMeas, ActivPress, ActiveMon'
    Args:
        station:

    Returns: df with station name, location, object type and fields listed above.
    '''

    ur1 = r'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getStationList&datasource=0&format=html&station_name='
    ur2 = r'&returnfields=station_name,station_no,site_name,station_latitude,station_longitude,object_type,ca_sta&ca_sta_returnfields='
    ur3 = r'Subbasin_Name,site_no,'
    ur4 = 'OwnerMon, MonSiteFreq, MonSGMASiteCode, MonSGMA, MonRMP, MonCASGEM, MonAgency, LastPressMeas, LastManMeas, ActivPress, ActiveMon,Depth_Category'.replace(
        ' ', '')
    url_send = ur1 + station + ur2 + ur3 + ur4
    content = helper.wiski_request_ssl(url_send)
    x = pd.read_html(content, header=0)
    x = x[0]
    x = x.loc[x.object_type.str.lower().str.contains('gw monitoring', na=False)]
    x = x.dropna(subset='station_name')
    # raw[column] = pd.to_datetime(raw[column], errors='coerce')  # Ensures proper datetime conversion
    x.loc[:, 'LastManMeas'] = pd.to_datetime(x.loc[:, 'LastManMeas'], errors='coerce')
    x.loc[:, 'LastPressMeas'] = pd.to_datetime(x.loc[:, 'LastPressMeas'], errors='coerce')

    x.loc[:, 'Depth_Category'] = x.loc[:, 'Depth_Category'].fillna('Other')

    x.loc[x.loc[:, 'Depth_Category'].str.contains('Deep'), 'Depth_Category'] = "Deep(>500ft)"

    x = x.astype({'LastPressMeas': 'datetime64[ns]',
                  'LastPressMeas': 'datetime64[ns]'})

    rep = "MonSGMA 	MonRMP 	MonCASGEM 	  	ActivPress 	ActiveMon".split()

    for c in rep:
        x.loc[:, c] = x.loc[:, c].fillna(False)
        x.loc[:, c] = x.loc[:, c].replace({'yes': True, 'no': False})

    x.loc[:, 'MonSGMASiteCode'] = x.loc[:, 'MonSGMASiteCode'].fillna('')

    return x


def get_allstation_via_station_char(reload_from_wiski=False):
    if (not reload_from_wiski) and os.path.exists('allinfo.pickle'):
        print('loading allinfo from pickle')
        allinfo = pd.read_pickle('allinfo.pickle')

    else:
        print('loading allinfo from wiski')
        allinfo = pd.concat(
            [
                get_gw_mon_status('SRP*'),
                get_gw_mon_status('Son*'),
                get_gw_mon_status('PET*'),
                get_gw_mon_status('LRR*'),

            ]
        )
        # print(allinfo.head())
        # print(allinfo.MonAgency.unique())

        # allinfo.loc[:, 'RMP_MO_Deep'] = allinfo.loc[:, 'RMP_MO_Deep'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        # allinfo.loc[:, 'RMP_MO_Shallow'] = allinfo.loc[:, 'RMP_MO_Shallow'].apply(
        #     lambda x: pd.to_numeric(x, errors='coerce'))
        # # allinfo = allinfo.astype({"RMP_MO_Deep":np.float64, "RMP_MO_Shallow":np.float64}, errors = 'ignore')
        # allinfo.loc[:, 'RMP_Shallow'] = allinfo.loc[:, 'RMP_MO_Shallow'].notnull()
        # allinfo.loc[:, 'RMP_Deep'] = allinfo.loc[:, 'RMP_MO_Deep'].notnull()
        #
        # print(allinfo.loc[:, 'RMP_MO_Shallow'].unique())

        allinfo = allinfo.rename(columns={'station_name': 'Station Name'})
        allinfo.index = allinfo.loc[:, 'Station Name']

        allinfo = get_any_meas_date_info(allinfo)

        allinfo.to_pickle("allinfo.pickle")

    return allinfo


def get_man():
    print('loading manmeas from pickle')
    man = pd.read_pickle('allinfo.pickle')
    man.loc[:, 'LastManMeas'] = pd.to_datetime(man.loc[:, 'LastManMeas'], errors='coerce')
    man = man.loc[man.LastManMeas.notnull()]
    # print(man.dtypes)

    # man.loc[:, 'yearmin'] = pd.to_datetime(man.loc[:, 'from']).dt.year
    man.loc[:, 'yearmax'] = pd.to_datetime(man.loc[:, 'LastManMeas']).dt.year
    man = get_meas_date_info(man, 'LastManMeas')
    man.to_pickle('manmeas.pickle')
    return man


def get_press():
    print('loading manmeas from pickle')
    press = pd.read_pickle('allinfo.pickle')
    press.loc[:, 'LastPressMeas'] = pd.to_datetime(press.loc[:, 'LastPressMeas'], errors='coerce')
    press = press.loc[press.LastPressMeas.notnull()]

    # man.loc[:, 'yearmin'] = pd.to_datetime(man.loc[:, 'from']).dt.year
    press.loc[:, 'yearmax'] = pd.to_datetime(press.loc[:, 'LastPressMeas']).dt.year
    press = get_meas_date_info(press, 'LastPressMeas')
    press.to_pickle('press.pickle')
    return press


def get_meas_date_info(raw, column="LastManMeas"):
    # re-label
    # print(raw.dtypes)
    # print(raw.head())

    raw.loc[:, 'date'] = pd.to_datetime(raw.loc[:, column], errors='coerce')

    raw.loc[:, "Elapsed Time"] = (datetime.datetime.now() - raw.loc[:, 'date']).dt.days / 365
    raw.loc[:, "Elapsed Time"] = np.round(raw.loc[:, "Elapsed Time"], 1)
    raw.loc[:, "Number of Months Since Last Measurement"] = (
            (datetime.datetime.now() - raw.loc[:, 'date']).dt.days / 30)
    raw.loc[:, "Number of Months Since Last Measurement"] = np.round(
        raw.loc[:, "Number of Months Since Last Measurement"], 1)
    raw.loc[:, 'Last Measurement'] = raw.loc[:, 'date'].dt.strftime("%b-%Y")
    # raw.loc[:, 'First Measurement'] = raw.loc[:, 'from'].dt.strftime("%b-%Y")

    return raw

def get_any_meas_date_info(raw):
    # re-label
    # print(raw.dtypes)
    # print(raw.head())

    raw.loc[:, 'date1'] = pd.to_datetime(raw.loc[:, 'LastManMeas'], errors='coerce')
    raw.loc[:, 'date2'] = pd.to_datetime(raw.loc[:, 'LastPressMeas'], errors='coerce')

    raw.loc[:, "et1"] = (datetime.datetime.now() - raw.loc[:, 'date1']).dt.days / 30
    raw.loc[:, "et2"] = (datetime.datetime.now() - raw.loc[:, 'date2']).dt.days / 30

    raw.loc[:,'Last Measurement'] = raw.loc[:,['et1','et2']].min(axis=1)

    raw.loc[:, "Last Measurement"] = np.round(
        raw.loc[:, "Last Measurement"], 1)

    raw = raw.drop(columns = ['date1','date2','et1', 'et2'])

    return raw


## load for first time
allinfo = get_allstation_via_station_char(reload_from_wiski=True)
ts = get_ts()
man = get_man()
press = get_press()

print(allinfo.head())


# delete files to make it re-load
def remove():
    # os.remove('ts.pickle')
    # os.remove('manmeas.pickle')
    # os.remove('allinfo.pickle')
    # os.remove('press.pickle')
    print('done removing files')


def get_loc(name):
    loci = allinfo[allinfo.loc[:, 'Station Name'] == name].loc[:,
           ['Station Name', 'station_latitude', 'station_longitude']]

    return [[loci.at[name, 'station_latitude']], [loci.at[name, 'station_longitude']]]


all_options = {
    'Santa Rosa Plain': sorted(allinfo.query("site_no=='SRP'").loc[:, 'Station Name'].unique()),
    'Sonoma Valley': sorted(allinfo.query("site_no=='Son'").loc[:, 'Station Name'].unique()),
    'Petaluma Valley': sorted(allinfo.query("site_no=='PET'").loc[:, 'Station Name'].unique()),
    'Lower Russian River': sorted(allinfo.query("site_no=='LRR'").loc[:, 'Station Name'].unique())
}

app.layout = html.Div([
    html.H3("Waterlevel Hydrographs"),
    html.Div([
        dcc.Graph(id='mapbox', style={'width': '100%', 'height': '100vh'}),
    ], style={'width': '100%', 'display': 'inline-block'}),

    html.Div([

        html.Div([
            html.H5("Monitoring Agency"),
            dcc.Checklist(
                id="monagency",
                options=[
                    {"label": "Sonoma County Water Agency", "value": "Sonoma County Water Agency"},
                    {"label": "Department of Water Resources", "value": "Department of Water Resources"},
                    {"label": 'Sonoma Valley GSA', "value": 'Sonoma Valley GSA'},
                    {"label": 'Petaluma Valley GSA', "value": 'Petaluma Valley GSA'},
                    {"label": 'Santa Rosa Plain GSA', "value": 'Santa Rosa Plain GSA'},
                    {"label": 'Sonoma Resource Conservation District',
                     "value": 'Sonoma Resource Conservation District'},
                    {"label": 'Sonoma County PRMD', "value": 'Sonoma County PRMD'},
                    {"label": 'City of Petaluma', "value": 'City of Petaluma Public Works and Utilities Department'},
                    {"label": "All", "value": "All"},
                ],
                labelStyle={"display": "block"},
                value=["Sonoma County Water Agency"],
            ), ], style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
            html.H5("Measurement Type"),
            dcc.RadioItems(
                id="pressure",
                options=[
                    {"label": "Manual Measurement", "value": "man"},
                    {"label": "Pressure Transducer", "value": "press"},
                    {"label": "All", "value": "all"},
                ],
                # labelStyle={"display": "block"},
                value="all",
                # multi=False,
            ),

            html.Div([
                html.H5("RMP"),
                dcc.Dropdown(
                    id="check_rmp",
                    options=[
                        {"label": "RMP", "value": "RMP"},
                        # {"label": "RMP Deep", "value": "RMP_Deep"},
                        {"label": "Non-RMP", "value": "Non-RMP"},
                        {"label": "All", "value": "All"},
                    ],

                    value="All",
                    multi=False,
                ),
            ],
                # style={ 'width' : '20%','display': 'inline-block', 'verticalAlign':'top'},
                # labelStyle={"display": "block"},
            )

        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H5("Actively Monitored"),

            dcc.RadioItems(
                id="activemon",
                options=[
                    {"label": "Actively Monitored", "value": 'active'},
                    {"label": "Not Actively Monitored", "value": 'inactive'},
                    {"label": "All", "value": "all"},
                ],
                # labelStyle={"display": "block"},
                value="active",
                # multi=False,
            ),

            html.H5("SGMA Well"),
            dcc.RadioItems(
                id="MonSGMA",
                options=[
                    {"label": "SGMA", "value": 'sgma'},
                    {"label": "Not SGMA", "value": 'nonsgmaa'},
                    {"label": "All", "value": "all"},
                ],
                # labelStyle={"display": "block"},
                value="all",
                # multi=False,
            ),

        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H5("Well Depth"),
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
            ), ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H5("Number of Months Since Last Observation"),
            dcc.Slider(
                value =1,
                min = 0,
                max = 18,
                marks = {m: f">{str(m)}m" if m in range(0,19,4) else '' for m in months},
                tooltip={"always_visible": False},
                # handleLabel={"showCurrentValue": True,"label": "Months"},
                vertical=False,
                # verticalHeight='100%',
                id="slider",
                # marks = ,
                # labelStyle={"display": "block"},
                # value=["Shallow (0-200ft)", "Medium (200-500ft)", "Deep(>500ft)"],
            ), ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    ],
        style={'width': '100%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Hr(),

        html.Div([
        html.Div([
            html.Button("Show Hydrograph(s)", id="show-image", n_clicks=0), ],
            style={'width': '20%', 'display': 'inline-block', }, ),


        dcc.Graph(id='graph'),

        html.Div([html.Div([
            html.Button("Download Data", id="download-data", n_clicks=0), ],
            style={'width': '20%', 'display': 'inline-block'}),
            dcc.Download(id="download-link"), ],
            style={'width': '100%', 'display': 'inline-block', }
        ),
    ])
])




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
        print('ff' * 30)
        print(colorscale)
        colorscale = [i['hovertext'] for i in colorscale['points']]
        print(colorscale)

    if len(colorscale) == 1:
        colorscale = colorscale[0]
        title = f"{colorscale}"
        print(f"the name of the station is {title}")
        x = wiski_data.wiski_plot(colorscale)

        x.get_station_pars(remove_pt=False)
        fig = x.plot_gw()

    else:
        print('plotting many' * 50)
        # asdfasdf
        x = wiski_data_plot_multi.wiski_plot(colorscale)

        fig = x.plot_multi()

    return fig


@callback(
    Output('mapbox', 'figure'),
    [
        # Input('mapbox', 'selectedData'),
        Input('checkbox', 'value'),
        Input('monagency', 'value'),
        # Input('depth-slider', 'value'),
        Input('check_rmp', 'value'),
        Input('activemon', 'value'),
        Input('MonSGMA', 'value'),
        Input('pressure', 'value'),
        # Input("show-map", "n_clicks"),
        Input("slider", "value"),
    ],
)
def update_figure(depth, monAgency, RMP_type, activemon, MonSGMA, pressure,
                  nmonths):  # Modify the function parameters
    # print(clicks)
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] != "show-map":
        print(ctx.triggered[0]["prop_id"])
        # print('preventing update')
        # raise PreventUpdate
    else:
        print(ctx.triggered[0]["prop_id"])
        # print('Allowing update')

    # allinfo = get_allstation()
    man = get_man()
    press = get_press()

    print(f"depth is of type {type(depth)}")
    print(f"this is the depth variable {depth}")
    if isinstance(depth, list):
        if 'all' in [x.lower() for x in depth]:
            cdf = allinfo
        else:
            # cdf = allinfo.query(f"Depth_Category=={depth}")
            # print(allinfo.Depth_Category.unique())
            cdf = allinfo.loc[allinfo.Depth_Category.isin(depth)]
    elif 'all' == depth.lower():
        cdf = allinfo
    else:
        cdf = allinfo.query(f"Depth_Category=={depth}")

    print(f"monAgency is of type {type(monAgency)}")
    print(f"this is the monAgency variable {monAgency}")
    if isinstance(monAgency, list):
        if 'all' in [x.lower() for x in monAgency]:
            cdf = cdf
        else:
            # cdf = allinfo.query(f"Depth_Category=={depth}")
            # print(allinfo.Depth_Category.unique())
            cdf = cdf.loc[cdf.MonAgency.isin(monAgency)]

    elif 'all' == monAgency.lower():
        cdf = cdf
    else:
        cdf = cdf.query(f"MonAgency=={monAgency}")

    print(f"activemon is of type {type(activemon)}")
    print(f"this is the activemon variable {activemon}")

    if 'all' == activemon.lower():
        cdf = cdf
    else:
        cdf = cdf.loc[cdf.ActiveMon == (activemon == 'active')]

    if 'all' == MonSGMA.lower():
        cdf = cdf
    else:
        cdf = cdf.loc[cdf.MonSGMA == (MonSGMA == 'sgma')]

    print(f"pressure is of type {type(pressure)}")
    print(f"this is the pressure variable {pressure}")
    if pressure.lower() == 'all':
        print('showing all')
        # Filter the data based on the year range
        ts_file = allinfo.copy()
    elif pressure.lower() == 'man':
        print('showing manual')
        ts_file = man.copy()
    else:  #assume it's pressure
        print('showing pressure')
        ts_file = press.copy()

    cdf = cdf.loc[cdf.loc[:, 'Station Name'].isin(ts_file.loc[:, 'Station Name'])]

    def convert_empty_strings_to_nan(data_frame, column_name):
        # Replace empty strings with NaN in the specified column
        data_frame[column_name] = data_frame[column_name].replace('', np.nan)
        return data_frame

    cdf = convert_empty_strings_to_nan(cdf, 'station_longitude')
    cdf = convert_empty_strings_to_nan(cdf, 'station_latitude')

    cdf = cdf.astype({'station_longitude': np.float64,
                      'station_latitude': np.float64}, errors='ignore')
    cdf = cdf.dropna(subset='station_longitude')

    if RMP_type == "RMP":
        cdf = cdf.loc[cdf.loc[:, 'MonRMP']]

    elif RMP_type == "Non-RMP":
        # print('sel non')
        # print(f"shape {cdf.shape}")
        # print(cdf.MonRMP.unique())
        cdf = cdf.loc[cdf.loc[:, 'MonRMP'] == False]
        # print(f"shape {cdf.shape}")

    # cdf.loc[cdf.loc[:, 'Station Name'].isin(ts_file.loc[:, 'station_name'])]

    print(cdf.loc[:, ['station_latitude', 'station_longitude']].describe())
    #
    marker_size = 2
    try:
        if pressure.lower() == 'all':
            print('makng simple map\n' * 5)
            cdf = cdf.loc[cdf.station_latitude.notnull()]
            cdf = cdf.loc[cdf.loc[:,'Last Measurement']>nmonths]
            cdf.loc[:, 'size'] = marker_size
            fig = px.scatter_mapbox(cdf,
                                    lat="station_latitude",
                                    lon="station_longitude",
                                    hover_name="Station Name",
                                    color='Depth_Category',
                                    size='size',

                                    hover_data={
                                        # "Elapsed Time": False,
                                        'station_no': True,
                                        "station_latitude": False,
                                        "station_longitude": False,
                                        "MonAgency": True,
                                        "size": False,
                                        "MonSGMASiteCode": True,
                                        'Last Measurement': True
                                    },
                                    )

        else:
            print(f"shape of ts_file is {ts_file.shape}")
            cur = ts_file.loc[ts_file.loc[:, 'Station Name'].isin(cdf.loc[:, 'Station Name'])]
            cur = cur.loc[cur.loc[:, 'Number of Months Since Last Measurement'] > nmonths]
            print(cur.loc[:, ['station_latitude', 'station_longitude']].describe())
            # cdf = pd.merge(cur, cdf.reset_index(drop = True), left_on= 'Station Name', right_on = "Station Name")
            #
            # cdf.loc[:,"Elapsed Time Min"] = cdf.loc[:, "Elapsed Time"].copy()
            # cdf.loc[cdf.loc[:,"Elapsed Time Min"]<3, "Elapsed Time Min"] = 3
            # print(cdf.loc[:,"Elapsed Time"].min())
            # print(cur.head().station_longitude.values)
            cur.loc[:, 'size'] = marker_size

            fig = px.scatter_mapbox(cur, lat="station_latitude",
                                    lon="station_longitude",
                                    hover_name="Station Name",
                                    # hover_data=["Elapsed Time",'station_no'],
                                    color='Number of Months Since Last Measurement',
                                    # size =  'Number of Months Since Last Measurement',
                                    # size_max= 15,
                                    size='size',
                                    hover_data={"Elapsed Time": True,
                                                'station_no': True,
                                                "station_latitude": False,
                                                "station_longitude": False,
                                                "MonSGMASiteCode": True,
                                                "MonAgency": True,
                                                'size':False,
                                                'Number of Months Since Last Measurement': True},
                                    )

            if cdf.shape[1] > 10:
                if "Source" in cdf.columns:
                    cdf = cdf.loc[:, :'Source']
                else:
                    cdf = cdf.iloc[:, :13]

            cdf.to_pickle('cur_selection.pickle')

        # Add mapbox style and enable scroll zoom
        fig.update_layout(
            mapbox_style="open-street-map",  # Example map style, you can change it
            mapbox_zoom=10,  # Adjust initial zoom level
            clickmode="event+select",
            mapbox_center={"lat": cdf['station_latitude'].mean(), "lon": cdf['station_longitude'].mean()},
        )

        fig.update_layout(
            dragmode="pan",  # Enables panning
            margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Remove margins for better view
            legend=dict(
                x=0.05,  # Position on the x-axis (0 to 1)
                y=0.95,  # Position on the y-axis (0 to 1)
                bgcolor="rgba(255,255,255,0.7)",  # Background color with transparency
                bordercolor="black",  # Border color
                borderwidth=15  # Border width
            ),

        )
        #
        # # Update layout to move the legend
        # fig.update_layout(
        #
        #     margin={"r": 0, "t": 0, "l": 0, "b": 0}  # Remove margins for better fit
        # )
        # # Enable scroll zoom explicitly
        # fig.show(config={'scrollZoom': True})

        print('the map figure does work')
    except Exception as e:
        print(e)
        print('\n\nit DOES  fail\n\n')
        fig = go.Figure()

    return fig


@app.callback(
    Output("download-link", "data"),
    [Input("download-data", "n_clicks")],
    prevent_initial_call=True,
)
def download_data(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    if os.path.exists("cur_selection.pickle"):
        filtered_data = pd.read_pickle("cur_selection.pickle")
    else:
        filtered_data = pd.DataFrame()

    # Generate CSV content based on filtered data
    csv_string = filtered_data.to_csv(index=False, encoding='utf-8')

    return dict(content=csv_string, filename='filtered_data.csv')


if __name__ == '__main__':
    app.run_server(debug=True, port=8056)
