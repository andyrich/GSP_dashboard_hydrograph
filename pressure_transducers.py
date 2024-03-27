import geopandas as gpd
import helper
import pandas as pd
import numpy as np
import dash
import datetime
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import os
# dash.register_page(__name__)

k = helper.get_kiwis()

def get_recent_measurements():

    '''
    get map with time since last measurement for all stations with pressure traansducer data
    Returns:

    '''

    print("loading Pressure transducers")
    par = k.get_timeseries_list(stationparameter_name="Pressure", return_fields=['station_name',  'station_no', 'coverage', 'ts_name'])

    raw = par.query("ts_name == '01.Raw-FeetH2O'")
    raw.loc[:, 'to'] = pd.to_datetime(raw.loc[:, 'to']).dt.tz_convert(None)
    raw.loc[:, 'from'] = pd.to_datetime(raw.loc[:, 'from']).dt.tz_convert(None)
    raw = raw.dropna(subset='to')

    raw = raw.astype({'to': 'datetime64[ns]',
                      'from': 'datetime64[ns]'})

    # re-label
    raw.loc[:, "Elapsed Time"] = (datetime.datetime.now() - raw.loc[:, 'to']).dt.days / 365
    raw.loc[:, "Elapsed Time"] = np.round(raw.loc[:, "Elapsed Time"], 1)
    raw.loc[:, "Number of Months Since Last Measurement"] = (
                (datetime.datetime.now() - raw.loc[:, 'to']).dt.days / 30)
    raw.loc[:, "Number of Months Since Last Measurement"] = np.round(
        raw.loc[:, "Number of Months Since Last Measurement"], 1)
    raw.loc[:, 'Last Measurement'] = raw.loc[:, 'to'].dt.strftime("%b-%Y")
    raw.loc[:, 'First Measurement'] = raw.loc[:, 'from'].dt.strftime("%b-%Y")


    raw.loc[:, 'Param'] = "Pressure"
    raw.to_csv('assets/pressure_table.csv')
    # since = gpd.GeoDataFrame(since, geometry=gpd.points_from_xy(since.station_longitude, since.station_latitude),
    #                          crs=4326)

    return raw


def get_manual_measurements():

    '''
    get map with time since last measurement for all stations with pressure traansducer data
    Returns:

    '''

    print("loading manual meas")

    par =[ k.get_timeseries_list(parametertype_name="Depth_to_water", station_name = site,
                            ts_name = "02.ManualMeasurement.E",
                                return_fields=['station_name',  'station_no', 'coverage']) for site in ['Son*', "PET*", "SRP*", 'LRR*'] ]
    par = pd.concat(par)

    # raw = par.loc[par.ts_name.str.contains('Manual')]
    raw = par.drop_duplicates('station_name')
    raw.loc[:, 'to'] = pd.to_datetime(raw.loc[:, 'to']).dt.tz_convert(None)
    raw.loc[:, 'from'] = pd.to_datetime(raw.loc[:, 'from']).dt.tz_convert(None)
    raw = raw.dropna(subset='to')

    raw = raw.astype({'to': 'datetime64[ns]',
                      'from': 'datetime64[ns]'})

    # re-label
    raw.loc[:, "Elapsed Time"] = (datetime.datetime.now() - raw.loc[:, 'to']).dt.days / 365
    raw.loc[:, "Elapsed Time"] = np.round(raw.loc[:, "Elapsed Time"], 1)
    raw.loc[:, "Number of Months Since Last Measurement"] = (
                (datetime.datetime.now() - raw.loc[:, 'to']).dt.days / 30)
    raw.loc[:, "Number of Months Since Last Measurement"] = np.round(
        raw.loc[:, "Number of Months Since Last Measurement"], 1)
    raw.loc[:, 'Last Measurement'] = raw.loc[:, 'to'].dt.strftime("%b-%Y")
    raw.loc[:, 'First Measurement'] = raw.loc[:, 'from'].dt.strftime("%b-%Y")

    # since = pd.merge(lon, raw)
    raw.loc[:,'Param'] = "Manual"
    raw.to_csv('assets/manual_table.csv')
    # since = gpd.GeoDataFrame(since, geometry=gpd.points_from_xy(since.station_longitude, since.station_latitude),
    #                          crs=4326)

    return raw

def get_all():
    df1 = get_recent_measurements()
    df2 = get_manual_measurements()

    lon = k.get_station_list(
        return_fields=['station_name', 'station_longitude', 'station_latitude'])

    raw = pd.concat([df1, df2])
    x = pd.to_numeric(lon.station_latitude, errors='coerce').notnull()
    lon = lon.loc[x]

    since = pd.merge(lon, raw, on = 'station_name')

    return  since

def plot_pressure_map(since = None):

    if since is None:
        since = get_recent_measurements()

    # m = since.drop(columns=['from', 'to']).explore('Number of Months Since Last Measurement',
    #                                            style_kwds={"style_function": lambda x: {
    #                                                "radius": x["properties"]["Elapsed Time"]}})
    #
    # m.save('assets/pressure.html')



def layout():
    # if os.path.exists('assets/pressure_table.csv'):
    #     df = pd.read_csv('assets/pressure_table.csv')
    #     df = df.sort_values('Number of Months Since Last Measurement', ascending=False)
    #     df = df.loc[:,['station_name','station_no',"Number of Months Since Last Measurement",
    #                     'Last Measurement','First Measurement']]
    # else:
    #
    #     # df = pd.DataFrame(columns = ['station_name','station_no',"Number of Months Since Last Measurement",
    #     #                 'Last Measurement','First Measurement'])

    df = get_recent_measurements()
    plot_pressure_map(df)
    print(df.head())
    return df
    # return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    # return html.Div([
    #     html.Br(),
    #     html.Br(),
    #     html.Br(),
    #
    #
    #     html.Br(),
    #     dcc.Link("Map of Pressure Transducer Updated data", href="assets/pressure.html", target="_blank"),
    #
    #     html.Br(),
    #     html.Div([
    #         dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    #     ], id = 'interactive-plot',
    #         style={'width': '100%', 'height': '100%', 'float': 'middle', 'display': 'block'}),
    #     ])