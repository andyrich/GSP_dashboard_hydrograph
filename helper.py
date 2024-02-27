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