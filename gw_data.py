import pandas as pd
import helper

import numpy as np
from datetime import date


def get_gw_data(pname, end_yr=2025, start_year=1900, resampled=False, keep_qc=False):

    '''
    pname may class ts_id. ts_id must be ana single string or integer this can come from above or using:
    t = gw.wiski_plot('Son0001')
    t.get_station_pars()
    if pname does not have ts_id attribute, then assume that pname=ts_id

    pname may also have pname.Param_reclass as attribute. this is the name of the df. if not present name will be 'Measurement'

    keep_qc = if True will keep only high-quality measurements
    end_yr/start_year =
        start/ending of download periods. can be
        int (year),
        list (year, month, day),
        pandas datetime,
        or None (year=1900 or 2025)
    '''

    if end_yr is int:
        to_date = date(end_yr, 1, 1)
    elif isinstance(end_yr, list):
        to_date = date(end_yr[0], end_yr[1], end_yr[2])
    elif isinstance(end_yr, pd.Timestamp):
        to_date = date(end_yr.year, end_yr.month, end_yr.day)
    else:
        to_date = date(2025, 1, 1)

    if start_year is int:
        start_date = date(start_year, 1, 1)
    elif isinstance(start_year, list):
        start_date = date(start_year[0], start_year[1], start_year[2])
    elif isinstance(start_year, pd.Timestamp):
        start_date = date(start_year.year, start_year.month, start_year.day)
    else:
        start_date = date(1900, 1, 1)

    if hasattr(pname, 'ts_id'):
        ts_id = int(pname.ts_id)
    else:
        ts_id = int(pname)

    return_fields = ['Timestamp', 'Value', "Quality Code"]

    k = helper.get_kiwis()
    f = k.get_timeseries_values(return_fields=return_fields, ts_id=ts_id, to=to_date,
                                **{'from': start_date}, keep_tz=False)

    f = f.dropna()
    f.index = f.index.tz_convert("US/Pacific").tz_localize(None)
    # f.index = f.index - pd.Timedelta('8hours')

    # filter measurements with quality code == 200 (unknown) and those with <80 (which are good, excellent)
    filt_bad = np.logical_or(f.loc[:, 'Quality Code'] == 200, f.loc[:, 'Quality Code'] < 80).values

    if keep_qc:
        return f, None
    else:
        bad_meas = f[~filt_bad].drop(columns=['Quality Code'])

    f = f.loc[filt_bad, :].drop(columns=['Quality Code'])

    if resampled:
        res = f.resample('1d').mean()
        res = res[res.Value.notnull()]
        res = res[~(res.Value > 2000)]
        res = res[~(res.Value < -250)]
        f = res.copy()

    if hasattr(pname, 'Param_reclass'):
        if isinstance(pname, pd.Series):
            rename = pname.Param_reclass
        elif isinstance(pname, pd.DataFrame):
            rename = pname.Param_reclass.unique()[0]
        elif isinstance(pname.Param_reclass, str):
            rename = pname.Param_reclass
        else:
            rename = 'Measurement'
    else:
        rename = 'Measurement'

    f = f.rename(columns={'Value': rename})

    return f, bad_meas


def get_gw_stations_in_basin(basins=None, final_only=True, basic_only = False, day_mean = False):
    '''
    use this to load data for running gwplot_wiski
    :param day_mean:
    :param basic_only:
    :param basins: list of station names with wildcards - eg 'Son*'
    :param final_only:  if True will filter to final measurements only

    :return: dataframe of stations with gw level data with station id's, parameter id's, from/to etc.
    '''
    if basins is None:
        basins = ['SRP*', 'Son*', 'PET*']
    import re

    if basic_only:
        url_blank = r'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getTimeseriesList&datasource=0&format=html&station_name={:}&stationparameter_name=*levation*&returnfields=station_name,station_no'
    else:
        url_blank = r'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getTimeseriesList&datasource=0&format=html&station_name={:}&stationparameter_name=*levation*&returnfields=station_name,station_no,ts_id,ts_name,parametertype_name,coverage,stationparameter_longname,stationparameter_no,ts_shortname,ts_type_name,parametertype_id,ts_id,ts_unitname'

    # https://www.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getrequestinfo

    assert isinstance(basins, list), f"basins needs to be a list, not {type(basins)}"

    #load basin using above url and request it
    # af.set_proxy()
    gw_stats = pd.DataFrame()
    for bas in basins:
        print(f"loading {bas}")
        url_in = url_blank.format(bas)

        gw_ = dload(url_in)
        if gw_.shape[0] > 0:
            # gw_stats = gw_stats.append(gw_)
            gw_stats = pd.concat([gw_stats, gw_])

    gw_stats = gw_stats.dropna(how='all')
    gw_stats.loc[:, 'ts_id'] = gw_stats.loc[:, 'ts_id'].fillna(-999)
    gw_stats = gw_stats.astype({'ts_id': int})
    gw_stats.loc[:, 'from'] = pd.to_datetime(gw_stats.loc[:, 'from']).dt.tz_localize(None)-pd.to_timedelta(8, unit = "H")
    gw_stats.loc[:, 'to'] = pd.to_datetime(gw_stats.loc[:, 'to']).dt.tz_localize(None)-pd.to_timedelta(8, unit = "H")


    if final_only:
        v_ = gw_stats.stationparameter_longname.str.upper().str.contains('ELEVATION').to_frame()

        fil = pd.concat([v_], axis=1)
        gw_elev = gw_stats.loc[fil.any(axis=1),:]

        gw_elev = gw_elev.loc[~gw_elev.ts_name.str.upper().str.contains('TEST'),:]
        gw_elev = gw_elev.loc[~gw_elev.ts_name.str.contains('(rel/abs)'),:]
        gw_elev = gw_elev.loc[~gw_elev.ts_name.str.contains('Offset'),:]

        gw_elev = gw_elev.loc[~gw_elev.ts_shortname.str.contains('Cmd.O'),:]

        gw_elev = gw_elev.loc[~gw_elev.parametertype_name.str.contains('Vapor'),:]

        # gw_elev= gw_elev.drop_duplicates(['ts_name','parametertype_name','stationparameter_longname','ts_shortname'])

        def recl(series):
            if any([(('PRESSURE' in x.upper()) | ('TRANSDUCER' in x.upper()) | ('DAY' in x.upper())
                     | ('MONTH' in x.upper())) for x in series.values]):
                return 'Pressure Transducer'
            else:
                return 'Manual Measurement'

        gw_elev['Param_reclass'] = gw_elev.loc[:, 'ts_name':'ts_type_name'].apply(recl, axis=1)

        gw_elev = gw_elev.loc[gw_elev['from'].notnull(),:]

        rep = lambda x: re.sub(r'\d+', '', x)
        gw_elev['Site'] = gw_elev['station_name'].apply(rep)

        gw_elev = gw_elev.sort_values('Site')


        return gw_elev
    else:
        return gw_stats

def dload(url):
    '''
    Try 3 times to load the url

    Args:
        url:

    Returns:

    '''
    for i in range(0, 4):
        print('trying...\n{:}'.format(url))
        try:
            content = helper.wiski_request_ssl(url)
            x = pd.read_html(content, header=0)
            gw_stats = x[0]
            print('success')
        except ImportError:
            print('fail', i)
            continue
        except HTTPError:
            print('failed with httperror', i)
            continue
        break

    return gw_stats