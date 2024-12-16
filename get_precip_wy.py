import pandas as pd
import helper
import requests
import io
ts_ids = {'Venado (Near Lake Sonoma)': "1964010",
          'Santa Rosa Airport': "1976010",
          'Ukiah Airport': '1977010',
          'Sonoma (General Vallejo)': "59424010"}


def get_precip(ts_i, raw = False):

    ts = ts_ids[ts_i]

    k = helper.get_kiwis()
    tab = k.get_timeseries_values(ts_id = ts, timezone = 'GMT-7',**{'from':'1/1/2014'},)

    if tab.shape[0] ==0:
        return None
    tab = tab.reset_index()
    tab.loc[:, 'Date'] = pd.to_datetime(tab.loc[:, 'Timestamp'])

    tab.loc[:, 'Date'] = pd.to_datetime(tab.loc[:, 'Date']).dt.tz_localize(None)
    # tab.loc[:, 'Date'] = pd.to_datetime(tab.loc[:, 'Date']).dt.tz_convert(None)
    if raw:
        tab = tab.set_index('Date', drop=False)
        # tab = tab.resample("1D").mean()
        tab.loc[:, 'wy'] = helper.water_year(tab.index)
        tab.loc[:, 'wy_date'] = helper.julian_water_year(tab.rename(columns = {'Date':'ogdate'}).reset_index().loc[:, 'Date'])
        return tab

    tab = tab.set_index('Date', drop=True)
    # tab.index = pd.to_datetime(tab.index).tz_convert("-08:00").tz_localize(None)

    tab = tab.resample("1D").mean()
    tab.loc[:, 'wy'] = helper.water_year(tab.index)
    tab.loc[:, 'wy_date'] = helper.julian_water_year(tab.reset_index().loc[:, 'Date'])

    #filter data to wy after 2015 or those with more than 250 observations per year
    cnt = tab.groupby('wy').count()
    cnt = cnt.loc[:,'Value']
    filt = (cnt>250) | (cnt.index>2015)
    cnt = cnt[filt]
    tab = tab.loc[tab.wy.isin(cnt.index),:]

    return tab



def get_cur_station(stat = 'Santa Rosa'):
    __df = get_precip(stat)
    if __df is None:
        return None

    __df = __df.set_index('wy_date')
    df_tot = pd.DataFrame()
    for g, dfi in __df.groupby('wy'):
        dfall_ci = dfi.loc[:,["Value"]].sort_values('wy_date').cumsum()
        dfall_ci.loc[:, 'wy'] = g
        # df_tot = df_tot.append(dfall_ci)
        df_tot = pd.concat([df_tot, dfall_ci])
    __df = df_tot.reset_index(drop = False)

    return __df

def get_allstations(options):
    if options is None:
        options = list(ts_ids.keys())
    ogdict = {key:get_cur_station(key) for key in options}

    #remove stations witout any data
    __dfall = {k: v for k, v in ogdict.items() if v is not None}

    return __dfall


def get_group(station, dfall_i123):
    return dfall_i123[station]

def get_station_min_max(df):
    # get min and max years
    dfstats = df[df.wy_date.dt.month==9].groupby('wy').max().sort_values('Value')
    xmin = dfstats.index.values[:2]
    xmind = {f'{xmin[0]} - Driest':xmin[0] }
    xmind[f'{xmin[1]} - Second Driest'] = xmin[1]
    # xmind[f'{xmin[2]} - Third Driest '] =  xmin[2]

    dfstats = df[df.wy_date.dt.month==9].groupby('wy').max().sort_values('Value', ascending=False)
    xmax = dfstats.index.values[:2]
    xmaxd = {f'{xmax[0]} - Wettest':xmax[0] }
    xmaxd[f'{xmax[1]} - Second Wettest'] = xmax[1]
    # xmaxd[f'{xmax[2]} - Third Wettest'] = xmax[2]

    extremes = pd.Series(xmin)
    # extremes = extremes.append(pd.Series(xmax))
    extremes = pd.concat([extremes, pd.Series(xmax)])

    return xmind, xmaxd, extremes