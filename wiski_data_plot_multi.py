# coding: utf-8

# from . wiski.wiski import dload, get_ts_data, get_gw_stations_in_basin, \
#     get_gw_stations_wellmeta_in_basin, setup_folder_for_plotting, station_plot_check
import datetime

import matplotlib.pyplot as plt
import plotly
from kiwis_pie import KIWIS
import numpy as np
import pandas as pd
import gw_data
import helper
import plotly.express as px
import plotly.graph_objects as go
# from . import arich_functions as af
import plot_help
# # import plot_wet as pw
# from . import plot_wet as pw
# from . import wiski
# from .wiski import gw_data
import warnings

#     get_ipython().magic(u'matplotlib inline')
class wiski_plot(object):
    '''

    This has only been setup to plot groundwater data from wiski, but could be expanded to import other parameters

    can simply be loaded with a station name, eg:
    t = gw.wiski_plot('Son0001')
    t.get_station_pars()
    ax = t.plot_gw(True, 'SRP')

    or you can specify df of parameters to download, but this df must come from running the following:
    url_son = r'https://www.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getTimeseriesList&datasource=0&format=html&station_name=\
    Son*&returnfields=station_name,station_no,ts_id,ts_name,\
    parametertype_name,coverage,stationparameter_longname,ts_shortname,ts_type_name'
    x= pd.read_html(url_son, header =0)
    gw_stats = x[0]
    fil = pd.concat( [v_],axis=1)
    gw_elev = gw_stats[fil.any(axis=1)]


    gw_elev = gw_elev[~gw_elev.ts_name.str.upper().str.contains('TEST')]
    gw_elev = gw_elev[~gw_elev.ts_name.str.contains('(rel/abs)')]
    gw_elev = gw_elev[~gw_elev.ts_name.str.contains('Offset')]

    gw_elev = gw_elev[~gw_elev.ts_shortname.str.contains('Cmd.O')]


    gw_elev = gw_elev[~gw_elev.parametertype_name.str.contains('Vapor')]

    # gw_elev= gw_elev.drop_duplicates(['ts_name','parametertype_name','stationparameter_longname','ts_shortname'])



    def recl(series):
        if any([(('PRESSURE' in x.upper()) |('TRANSDUCER' in x.upper() )) for x in series.values]):
            return 'Pressure Transducer'
        else:
            return 'Manual Measurement'
    gw_elev['Param_reclass'] = gw_elev.loc[:,'ts_name':'ts_type_name'].apply(recl, axis=1)


    gw_elev = gw_elev.rename(columns = {'station_name':'Station Name', 'station_no':'Station Number'})

    example:
    x = wiski_plot('SRP0723')

    x.plot_gw()

    plt.show()

    '''

    def __init__(self, station_list, pars=None):

        self.station_list = station_list
        self.yearend = pars
        self.kurl = 'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?'
        self.k = KIWIS(self.kurl)
        # af.set_proxy()

        self.gw_elev = None
        self.pressure_trans = None
        self.bad_meas = None
        self.manual_meas = None

    def get_station_pars(self,station, remove_pt=True):
        '''
        GET GW pars for one station
        '''
        print('importing the parameters for {:}'.format(station))
        url_srp1 = r'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getTimeseriesList&datasource=0&format=html&station_name='
        url_srp2 = r'&returnfields=station_name,station_no,ts_id,ts_name,parametertype_name,coverage,stationparameter_longname,ts_shortname,ts_type_name'

        url_send = url_srp1 + station + url_srp2
        # content = wiski.wiski.wiski_request_ssl(url_send)
        content = helper.wiski_request_ssl(url_send)
        x = pd.read_html(content, header=0)
        gw_stats = x[0]
        gw_stats = gw_stats.dropna(subset = 'from')

        print(gw_stats)
        self.RMP_ts = gw_stats.query("ts_name.str.contains('RMP')")

        # filter to only depth to water and pressure measurements, then keep only one of these per station
        v_ = gw_stats.stationparameter_longname.str.upper().str.contains('ELEVATION').to_frame()
        fil = pd.concat([v_], axis=1)
        gw_elev = gw_stats[fil.any(axis=1)]
        gw_elev = gw_elev.loc[~gw_elev.ts_name.str.upper().str.contains('TEST'), :]
        gw_elev = gw_elev.loc[~gw_elev.ts_name.str.contains('(rel/abs)'), :]
        gw_elev = gw_elev.loc[~gw_elev.ts_name.str.contains('Offset'), :]
        gw_elev = gw_elev.loc[~gw_elev.ts_name.str.contains('.ManualMeasurement.'),
                  :]  # for caissons which are calculated differently
        # gw_elev = gw_elev.loc[~gw_elev.ts_shortname.str.contains('Cmd.O'),:]
        gw_elev = gw_elev.loc[~gw_elev.parametertype_name.str.contains('Vapor'), :]
        gw_elev = gw_elev.loc[~gw_elev.ts_shortname.str.contains('Year'), :]  # drop stupid columns morgan added
        gw_elev = gw_elev.loc[~gw_elev.ts_shortname.str.contains('year'), :]  # drop stupid columns morgan added
        gw_elev = gw_elev.loc[~gw_elev.ts_shortname.str.contains('Max'), :]  # drop stupid columns morgan added

        # if there is dailymean calculated for station, then remove the raw transducer
        if (gw_elev.ts_shortname.str.contains('day|daily', regex=True, case=False).sum() > 0) or remove_pt:
            print('removing raw pressure transducer data because there is a day/daily mean value present for station')
            gw_elev = gw_elev.loc[~gw_elev.ts_name.str.lower().str.contains('transducer'), :]

        def recl(series):
            if any([(('PRESSURE' in x.upper()) | ('TRANSDUCER' in x.upper())) | ('DAY.MEAN' in x.upper())
                    | ('DAY MEAN' in x.upper()) | ('DAILYMEAN' in x.upper()) |
                    ('DAILY MEAN' in x.upper()) for x in series.values]):
                return 'Pressure Transducer'
            else:
                return 'Manual Measurement'

        # do re-class of timeseries names
        gw_elev.loc[:, 'Param_reclass'] = gw_elev.loc[:, 'ts_name':'ts_type_name'].apply(recl, axis=1)


        # finally classify all "DayMeanEdit" as Daily Pressure Transducer
        c = gw_elev.loc[:, 'ts_name'] == 'DayMeanEdit'
        gw_elev.loc[c, 'Param_reclass'] = 'Pressure Transducer'

        # remove reclassified pressure transducer data
        # need to re-do again b/c some names weren't caught in the recl function
        # if remove_pt:
        c = gw_elev.loc[:, 'ts_name'] == 'Day.Mean.Production'
        gw_elev = gw_elev.loc[~c, :]

        # drop weird timeseries (the monthly/yearly averages)
        c = gw_elev.ts_shortname.str.contains('spring|month|year|fall', regex=True, case=False)
        gw_elev = gw_elev.loc[~c, :]
        print(f'post-filter \n{gw_elev}')
        # c = gw_elev.ts_shortname.str.contains('spring|month|year|fall', regex=True, case=False)
        # gw_elev = gw_elev.loc[~c,:]

        gw_elev = gw_elev.rename(columns={'station_name': 'Station Name', 'station_no': 'Station Number'})
        gw_elev = gw_elev.loc[gw_elev['from'].notnull(), :]
        gw_elev.loc[:, 'from'] = pd.to_datetime(gw_elev.loc[:, 'from']).dt.tz_localize(None)
        gw_elev.loc[:, 'to'] = pd.to_datetime(gw_elev.loc[:, 'to']).dt.tz_localize(None)
        self.gw_elev = gw_elev

    def plot_multi(self, plot_wet =True):
        fig = go.Figure()

        if plot_wet:
            helper.do_plot_wet(fig)

        for num, stat in enumerate(self.station_list):
            try:
                self.get_station_pars(remove_pt=False, station=stat)
                fig = self.plot_gw(stat,number = num, fig = fig)
            except Exception as e:
                print(f"failed {stat} because {e}")

        return fig

    def plot_gw(self, station, seasonal=False, number = None,
                limit_trans2manual_dates=True, remove_pt=False, xlims=None,  fig = None):
        '''
        plot_wet = plot backgorund wet/dry (not implemented)
        gw_basin = ['SRP','SON','PET'] basin from which wet/dry record will be plot
        seasonal = to take the seasonal value to plot, ie spring and fall
        plot_dry =
        xlims = [yearsstart_plot, year_end_plot] or None. if None will be found by data limits. (default None)
        limit_trans2manual_dates = limit the pressure transducers to the beginning/end of the manual dates becaus
        of bad data in PT data
        y_axis_range_min = min range for hydrograph
        '''



        if self.gw_elev is None:
            print(f'getting parameter info for the station {station}')
            self.get_station_pars(remove_pt=remove_pt)

        self.check_if_no_gw_data()


        if fig is None:
            fig = go.Figure()

        if self.gw_data_empty:
            return fig

        color = get_plotly_colors_discrete(len(self.station_list))[number]


        alldat = pd.DataFrame()
        for _, pname in self.gw_elev.sort_values('Param_reclass',ascending = False).iterrows():
            # f, bad_meas = self.get_gw_data(pname, start_year=1900)
            print(f"\n\nloading the following\n:{pname}\n\n")
            f, bad_meas = gw_data.get_gw_data(pname, start_year=1900)
            print('done loading\n\n')
            # change marker/plot type depending on if manual measurement
            if 'Manual' in pname.Param_reclass and f.shape[0]<365*2:
                style = 's'
                self.manual_meas = f

            else:
                style = '-'
                if limit_trans2manual_dates:
                    if self.gw_elev.loc[self.gw_elev.ts_name.str.contains('Manual'), 'from'].shape[0] > 0:
                        manual_min = pd.to_datetime(
                            self.gw_elev.loc[self.gw_elev.ts_name.str.contains('Manual'), 'from'].values[0])
                        manual_max = pd.to_datetime(
                            self.gw_elev.loc[self.gw_elev.ts_name.str.contains('Manual'), 'to'].values[0])
                        f = f.loc[manual_min:manual_max, :]

                    print('limiting pressure transducers data to limit of manual measurement')

                self.pressure_trans = f

            if f.shape[0] > 0:
                # plot the gw data
                fresh = f.resample("1D").mean()
                print(fresh.head())
                if style == '-':
                    if "Pressure Transducer" in fresh.columns:
                        fig.add_trace(go.Scatter(x=fresh.index.values, y=fresh.loc[:,'Pressure Transducer'],
                                                 mode='lines',
                                                 line = {'color':color},
                                                 name=pname['Param_reclass'].replace("_"," "),
                                      legendgroup= f"group{number}",
                                      legendgrouptitle_text=station,)
                                      )
                    elif "Manual Measurement" in fresh.columns: #should only occur when there are manual meas that are are actually PT that are mislabeld as manual meas
                        fig.add_trace(go.Scatter(x=fresh.index.values, y=fresh.loc[:,'Manual Measurement'],
                                                 mode='lines',
                                                 name="Pressure Transducer",
                                      legendgroup= f"group{number}",
                                      legendgrouptitle_text=station,)
                                      )

                        warnings.warn("'Pressure Transducer' is not in the columns plotting")
                else:
                    marker = go.scatter.Marker(size=6, symbol='square', )
                    marker.color = color
                    fig.add_trace(go.Scatter(x=fresh.index.values, y=fresh.loc[:,'Manual Measurement'],
                                             mode='markers', marker = marker ,
                                             fillcolor= 'black',
                                             name=pname['Param_reclass'].replace("_"," "),
                                             legendgroup=f"group{number}",
                                             legendgrouptitle_text=station, )
                    )

            # alldat = alldat.append(f)
            alldat = pd.concat([alldat, f])
            # plot fall/spring average for manual measurements only
            if seasonal and ('Manual' in pname.Param_reclass):
                spring, fall = self.seasonal_mean(f)
                # markerf = go.scatter.Marker(size=10, symbol='square')
                # markerf.color = 'orange'

                markerf = go.scatter.Marker(
                    size=10,
                    symbol='square',
                    color=color,  # Fill color
                    line=dict(color='orange', width=2)  # Outline color and thickness
                )

                markers = go.scatter.Marker(
                    size=10,
                    symbol='square',
                    color=color,  # Fill color
                    line=dict(color='green', width=2)  # Outline color and thickness
                )

                fig.add_trace(go.Scatter(x=fall.index, y=fall.loc[:,'Manual Measurement'],
                                        mode='markers',
                                        marker = markerf,
                                        name= 'Fall Average',
                                         legendgroup=f"group{number}",
                                         legendgrouptitle_text=station,
                                         ))

                fig.add_trace(go.Scatter(x=spring.index, y=spring.loc[:,'Manual Measurement'],
                                        mode='markers',
                                        marker=markers,
                                        name= 'Spring Average',
                                         legendgroup=f"group{number}",
                                         legendgrouptitle_text=station,
                                         ))

                # plot the lines connecting the seasonal measurements
                tot = pd.concat([fall, spring]).sort_index()
                tot['date'] = tot.index
                tot = tot.groupby(pd.Grouper(freq='2QS')).first()
                line = go.scatter.Line(color = 'grey')
                # if not RMP:
                #     fig.add_trace(go.Scatter(x=tot.date, y=tot.loc[:,'Manual Measurement'],
                #                              showlegend=False,
                #                           name= '' ,line = line,
                #                              legendgroup="group1",
                #                              legendgrouptitle_text="Groundwater Observations",
                #                              ))

                seasonal = False

            # plot the bad measurements if they exist
            if bad_meas.shape[0] != 0:

                markers = go.scatter.Marker(size=5, symbol='square')
                markers.color = 'red'

                fig.add_trace(go.Scatter(x=bad_meas.index, y=bad_meas.loc[:,'Value'],
                                    mode='markers',
                                    marker = markers,
                                    name= 'Questionable Measurement',

                                    legendgroup=f"group{number}",
                                    legendgrouptitle_text=station,
                                         ))

                self.bad_meas = bad_meas
            else:
                self.bad_meas = None


        # if no data has been plotted, this will cause it to fail
        if hasattr(alldat.index.min(), 'year'):
            yearstart_plot = np.min([(alldat.index.min().year // 5) * 5, 2000])
        else:
            yearstart_plot = 2000

        if xlims is None:
            year_end_plot = None
        else:
            yearstart_plot = xlims[0]
            year_end_plot = xlims[1]

        fig.update_layout(xaxis=dict(range=[datetime.date(yearstart_plot,1,1),
                                            datetime.date(datetime.datetime.now().year+1,1,1),]))

        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",


        template='plotly',
        title=dict(
            text=', '.join(self.station_list),
            font=dict(size=24, color='black'),
            x=0.5,
            y=0.9
        ),
        xaxis_title=dict(text='Date', font=dict(size=16, color='#FFFFFF')),
        yaxis_title=dict(text='Groundwater Elevation (feet)', font=dict(size=16, color='black')),
        plot_bgcolor='white',
        xaxis=dict(tickfont=dict(size=14, color='black')),
        yaxis=dict(tickfont=dict(size=14, color='black')),
        legend=dict( y=1.1, orientation='h', font=dict(color='black')),
        margin=dict(l=10, r=10, t=100, b=50)
        )
        fig.update_xaxes(title_font_family="Arial")
        fig.layout.yaxis.gridcolor = 'grey'
        fig.layout.xaxis.gridcolor = 'black'

        fig.update_layout(
            height = 800,

            legend=dict(groupclick="toggleitem",
                                      y=-.1, orientation='h', font=dict(color='black')))

        return fig

    def get_well_info(self):

        ur1 = r'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?service=kisters&type=queryServices&request=getStationList&datasource=0&format=html&station_name='
        ur2 = r'&returnfields=station_name,station_no,site_name,station_latitude,station_longitude,ca_sta&ca_sta_returnfields=Vert_coord_Loc,TOS_1,BOS_1,TOS_2,BOS_2,TOS_3,BOS_3,TOS_4,BOS_4,TOS_5,BOS_5,TOS_6,BOS_6,TOS_7,BOS_7,TOS_8,BOS_8,'
        ur3 = r'Depth_to_Bot_Perf,Depth_to_Top_Perf,Web_GW_Obs_Range,Subbasin_Name,GWREF_DATUM,Web_Station_Name,Well_Depth'

        url_send = ur1 + self.station + ur2 + ur3
        content = helper.wiski_request_ssl(url_send)
        x = pd.read_html(content, header=0)

        return x[0].loc[0]

    def seasonal_mean(self, df, grouped=False):
        def sean(df):
            spring = np.logical_and((df.index.month >= 3), (df.index.month <= 5))
            fall = np.logical_and((df.index.month >= 10), (df.index.month <= 12))

            def _d(filt):
                temp = df[filt]
                temp.loc[:, 'date'] = pd.to_numeric(temp.index)
                # temp['date'] = pd.to_numeric(temp.index)
                temp.index = temp.index.year

                grouped = temp.groupby([temp.index]).mean()
                grouped.date = pd.to_datetime(grouped.date)
                grouped = grouped.set_index('date')
                return grouped

            return _d(spring), _d(fall)

        spring, fall = sean(df)

        if grouped is False:
            return spring, fall
        else:

            # tot = fall.append(spring).sort_index()
            tot = pd.concat([fall, spring]).sort_index()
            tot['date'] = tot.index
            tot = tot.groupby(pd.Grouper(freq='2QS')).first()
            return tot

        # fall.columns,spring.columns = ['Average Fall Measurement'],['Average Spring Measurement'



    def check_if_no_gw_data(self):
        '''
        check if there's not gw data after loading params. if None then skip plotting
        Args:
            self:

        Returns:

        '''
        if self.gw_elev.shape[0] == 0:
            # self.ax.set_title(self.station)
            self.gw_data_empty = True
        else:
            self.gw_data_empty = False

def get_plotly_colors_discrete(n):
    palette = plotly.colors.DEFAULT_PLOTLY_COLORS
    return [palette[i % len(palette)] for i in range(n)]