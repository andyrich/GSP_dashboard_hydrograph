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
    f.index = f.index.tz_localize(None)
    f.index = f.index - pd.Timedelta('8hours')

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
        res = res[~(res.Value < -200)]
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
