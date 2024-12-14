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

    # print(df)
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

def do_plot_wet(fig):
    dfwet = pd.read_csv('SRP_SON_PET_water_types.csv', index_col=[0])

    dfwet = dfwet.rename(columns={"WY_TYPE": "Type", 'wy.1': "WY"})
    dfwet = dfwet.set_index('WY')
    # print(dfwet)

    # print(dfwet.head())
    colors = {'Very Wet': 'cornflowerblue',
              'Wet': "lightblue",
              'Dry': "palegoldenrod",
              "Very Dry": "gold"}
    # for year in np.arange(2000,2018, 3):
    dfwet = dfwet.loc[dfwet.loc[:, 'Type'] != 'Normal']

    wytext = []

    for wy, row in dfwet.iterrows():
        ys = datetime.datetime(wy, 1, 1)
        ye = datetime.datetime(wy + 1, 1, 1)
        if row['Type'] in wytext:
            fig.add_vrect(x0=ys, x1=ye, line_width=0, fillcolor=colors[row['Type']],
                          layer="below",
                          opacity=0.8,
                          legendgroup="precip",
                          legendgrouptitle_text="Water Year Type",
                          name=row['Type'],
                          showlegend=False
                          )
        else:
            fig.add_vrect(x0=ys, x1=ye, line_width=0, fillcolor=colors[row['Type']],
                          layer="below",
                          opacity=0.8,
                          legendgroup="precip",
                          legendgrouptitle_text="Water Year Type",
                          name=row['Type'],
                          showlegend=True
                          )

        wytext.extend([row['Type']])

    return wytext


def download_daily(station, begin_year=1900, begin_month=10, param="00060"):
    '''
    download daily streamflow data from usgs
    for info see: https://waterservices.usgs.gov/rest/DV-Test-Tool.html
    Args:
        station: str (eg '11458500')
        begin_year:
        param: discharge: "00060_00003"  stage: 00065_00003"

    Returns: flow, info

    '''

    # url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={:}&referred_module=sw&period=&begin_date={:}-{:}-01'.\
    # format(station, begin_year, begin_month)
    url = 'https://waterdata.usgs.gov/nwis/dv?cb_{:}=on&format=rdb&site_no={:}&referred_module=sw&period=&begin_date={:}-{:}-01'. \
        format(param, station, begin_year, begin_month)
    top = pd.read_csv(url,
                      comment='#', sep='\t+', parse_dates=['datetime'], infer_datetime_format=True, )
    col = [x for x in top.columns if x.endswith(param + "_00003")]
    print(col)
    top = top.rename(columns={col[0]: 'Q'})

    top.index = pd.to_datetime(top.datetime, errors='coerce')
    top['Q'] = pd.to_numeric(top['Q'], errors='coerce')
    top = top.loc[:, ['Q']]
    top = top.dropna()

    info = pd.read_html(
        f"https://waterdata.usgs.gov/nwis/inventory?search_site_no={station}&search_site_no_match_type=exact&format=station_list&group_key=NONE&list_of_search_criteria=search_site_no")

    return top, info[1]



def water_year(date):
    '''
	this returns an integer water year of the date
	'''

    def wy(date):
        if date.month < 10:
            return date.year
        else:
            return date.year + 1

    if isinstance(date, pd.Series):
        return date.apply(wy)
    if isinstance(date, datetime.datetime):
        return wy(date)
    elif isinstance(date, pd.DatetimeIndex):
        return [wy(i) for i in date]
    else:
        import warnings
        warnings.warn('not a Series/datetime/DatetimeIndex object')
        # print('not a Series/datetime/DatetimeIndex object')
        return np.nan

def julian_water_year(wy):
    '''
    return days from start of water year, creates pseudo date from start of WY 2020.
    Args:
        date: datetimeindex, series, or dataframe

    Returns:

    '''
    if isinstance(wy, pd.DatetimeIndex):
        print('converting datetimeindex to df')
        wy.name = 'Date'
        wy = pd.DataFrame(wy, columns=['Date'])
    elif isinstance(wy, pd.Series):
        wy = wy.to_frame('Date')
    elif isinstance(wy, pd.DataFrame):
        print('using index from dataframe to get julian water year date')
        wy = wy.index
        wy.name = 'Date'
        wy = pd.DataFrame(wy, columns=['Date'])
    else:
        raise AssertionError(f"type is not supported {type(wy)}")

    if hasattr(wy, 'month'):
        wy.loc[:, 'month'] = wy.Date.month
        wy.loc[:, 'year'] = wy.Date.year
        wy.loc[:, 'day'] = wy.Date.day
    else:
        wy.loc[:, 'month'] = wy.Date.dt.month
        wy.loc[:, 'year'] = wy.Date.dt.year
        wy.loc[:, 'day'] = wy.Date.dt.day

    c = wy.loc[:, 'month'] < 10
    wy.loc[:, 'year'] = 2000
    wy.loc[~c, 'year'] = 1999
    wy.loc[:, 'WY_date'] = pd.to_datetime(wy.loc[:, ['year', 'month', 'day']])

    wy = wy.loc[:, 'WY_date'].values

    return wy