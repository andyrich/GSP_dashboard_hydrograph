import os
import requests
from kiwis_pie import KIWIS
from calendar import monthrange
import datetime
import pandas as pd
import numpy as np


def wiski_request_ssl(url):
    '''
    request data from wiski using only pandas
    :param url:
    :return: content to be read with pd.read_csv/read_html etc

    Examples:
    url = www.example.data.com
    content = wiski_request_ssl(url)
    df = pd.read_csv(content)

    '''

    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "kisters-net-chain.pem")
    content = requests.get(url, verify =DATA_PATH).content

    return content

def get_kiwis():
    '''
    get kiwis object
    Returns:

    '''

    kurl = 'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?'

    #add the certif file bc kiwis ssl file is problematic
    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "kisters-net-chain.pem")

    k = KIWIS(kurl, verify_ssl = DATA_PATH)

    return k

def load_wiski_rmp_value(station, isw = False):
    '''
    load the RMP.Fall and RMP.Spring gwl values from WISKI without calculating/loading raw data
    :param water_year:
    :param alt_station:
    :return: fall, spring, df, dfraw
    '''


    k = get_kiwis()
    date_start = datetime.datetime(2020, 1, 1)  # limit to just months necessary to inspect


    return_fields = ['station_name',
     'station_no',
     'station_id',
     'ts_id',
     'ts_name',
     'parametertype_id',
     'parametertype_name',
     'coverage']

    gwstats = k.get_timeseries_list(station_name=station, ts_name="RMP*",return_fields = return_fields)


    if gwstats.query("ts_name=='RMP.Fall' and to.str.contains('0')").shape[0] > 0:
        df_fall = k.get_timeseries_values(ts_id=gwstats.query("ts_name=='RMP.Fall'").loc[:, 'ts_id'],
                                       **{'from': date_start})

        df_fall.index = pd.to_datetime(df_fall.index).tz_convert("-08:00").tz_localize(None)

        df_fall.loc[:,'Month'] = df_fall.index.month
        if isw:
            df_fall = df_fall.loc[df_fall.Month== 8]
        else:
            df_fall = df_fall.loc[~(df_fall.Month == 8)]


    else:
        df_fall = pd.DataFrame([np.nan], index = [datetime.datetime(2020,1,1)], columns = ['Value'])


    if gwstats.query("ts_name=='RMP.Spring' and to.str.contains('0')").shape[0] > 0:
        df_spring = k.get_timeseries_values(ts_id=gwstats.query("ts_name=='RMP.Spring'").loc[:, 'ts_id'],
                                       **{'from': date_start})

        df_spring.index = pd.to_datetime(df_spring.index).tz_convert("-08:00").tz_localize(None)


        df_spring.loc[:,'Month'] = df_spring.index.month
        if isw:
            df_spring = df_spring.loc[df_spring.Month== 8]
        else:
            df_spring = df_spring.loc[~(df_spring.Month == 8)]


    else:
        df_spring = pd.DataFrame([np.nan], index = [datetime.datetime(2020,1,1)], columns = ['Value'])




    return df_fall, df_spring

def get_wl_smcs(stats = None, isw = False):
    '''
    load the actual waterlevel MT/MO/IM
    :param basin: site to search in
    :param stats: list (should be df, list or str) of station names or dataframe with series named station_name (ie from get_wl_smc_stations)
    :param isw: True/false if true will filter to columns containing isw. otherwise will be RMP (like all gwl MT/MO/s)
    :return: df of all waterlevel smcs get_wl_smc_stations
    '''
    url = "https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getStationList&format=html&station_name={:}&returnfields=station_name,station_no,ca_sta"


    if isinstance(stats, list) :
        stations = stats[0]
    elif isinstance(stats, str):
        stations = stats
    else:
        raise ValueError('not sure what input type stats is. should be df, list or str')


    url = url.format(stations)
    print(f"here are the stations being requested\n{stations}")

    df = wiski_request_ssl(url)
    df = pd.read_html(df, header=0)[0]

    if isw:
        df = df.set_index('station_name').filter(regex='ISW')
    else:
        df = df.set_index('station_name').filter(regex='RMP')

    df = df.dropna(axis = 0, how = 'all')

    df = df[df.index.notnull()].reset_index()

    if 'station_name' in df.columns:
        df = df.set_index('station_name')

    print(df)
    # calculate MT/MO's
    MT = df.filter(regex='MT').loc[[stations]].dropna(how='all', axis='columns')
    MT.columns = ['MT']
    MT = MT.at[stations, 'MT']

    MO = df.filter(regex='MO').loc[[stations]].dropna(how='all', axis='columns')
    MO.columns = ['MO']
    MO = MO.at[stations, 'MO']

    # IM = df.filter(regex='IM').loc[stations].dropna(how='all', axis='columns')
    # try:
    #     IM.columns = ['IM']
    #     IM = IM.at[stations, 'IM']
    # except:
    #     IM = ''

    return MT, MO

