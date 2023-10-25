from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import wiski_data
import gw_data
import helper
import numpy as np
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

# df = gw_data.get_gw_stations_in_basin()

k = helper.get_kiwis()
# df = k.get_station_list()

df2 = k.get_parameter_list(station_name = 'Son*')
# print(df2.parametertype_name.unique())
df2 = df2.query("parametertype_name=='Groundwater_elevation'")
df2 = df2.query("station_name.str.startswith('Son')")
# print(df2)

print(df2.head())

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    dcc.Dropdown(np.sort(df2.loc[:,'station_name'].unique()), df2.loc[:,'station_name'].unique()[0], id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    # dff = df[df.country==value]
    # return px.line(dff, x='year', y='pop')

    x =wiski_data.wiski_plot(value)
    x.get_station_pars(remove_pt=True)
    fig = x.plot_gw()

    return fig

if __name__ == '__main__':
    app.run(debug=True, port =8059)