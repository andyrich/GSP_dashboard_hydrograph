import helper
import plotly.express as px

def load_data():
    k = helper.get_kiwis()


    son_outflow = k.get_timeseries_values(ts_id=50966010, **{'from': '10/1/2014', 'timezone': "GMT+7"})
    son_outflow.index = helper.tz_fix(son_outflow)
    son_outflow = son_outflow.resample("1D").mean()

    son_outflow.loc[:, 'Water Year'] = helper.water_year(son_outflow)
    son_outflow.loc[:, 'Julian Date'] = helper.julian_water_year(son_outflow)

    son_outflow = son_outflow.reset_index()

    men_outflow = k.get_timeseries_values(ts_id=50966010, **{'from': '10/1/2014', 'timezone': "GMT+7"})
    men_outflow.index = helper.tz_fix(men_outflow)
    men_outflow = men_outflow.resample("1D").mean()

    men_outflow.loc[:, 'Water Year'] = helper.water_year(men_outflow)
    men_outflow.loc[:, 'Julian Date'] = helper.julian_water_year(men_outflow)
    men_outflow = men_outflow.reset_index()


    return men_outflow, son_outflow



def plot_timeseries_flows(x, title):

    fignew = px.line(x.reset_index(), x="Timestamp", y="Value",
                     # color='Water Year',
                     # width=1200, height=600,
                     title=title,
                     log_y=True,
                     labels={
                         "dy": "Day of Year",
                         "Value": "cfs",

                     }, )

    fignew.update_layout(hovermode="x")
    fignew.update_xaxes(tickformat="%y %b")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
    fignew.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 20},  # Remove margins for better view
    )
    return fignew


def plot_water_year_flows(x, title):
    fignew = px.line(x, x="Julian Date", y="Value", color='Water Year',
                     # width=1200, height=600,
                     title=title,
                     log_y=True,
                     labels={
                         "dy": "Day of Year",
                         "Value": "cfs",

                     }, )

    fignew.update_layout(hovermode="x")
    fignew.update_xaxes(tickformat="%b %d")
    fignew.update_traces(mode="lines", hovertemplate='%{y:,d} <i>af</i>')
    fignew.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 20},  # Remove margins for better view
    )

    yearmax = x.loc[:,'Water Year'].max()

    fignew.for_each_trace(
        lambda trace: trace.line.update( width=7) if trace.name == f"{yearmax}" else (),
    )

    return fignew