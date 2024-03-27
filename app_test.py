import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import dash_table
import pressure_transducers

'''
shows a map, hydrograph of pressure transducers
'''

# Sample DataFrame
df = pressure_transducers.layout()
df = df.astype({"station_longitude": float,
                "station_latitude": float})

# Rename latitude and longitude columns
df.rename(columns={'station_latitude': 'Station_Latitude', 'station_longitude': 'Station_Longitude',
                   'station_name': "Station_Name"}, inplace=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    dcc.Graph(id='map-graph'),
    html.Button('Generate Scatter Plot', id='scatter-button', n_clicks=0),
    dcc.Graph(id='scatter-plot'),
    dash_table.DataTable(
        id='datatable',
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.to_dict('records'),
        row_selectable='multi'
    )
])


# Define callback to update map based on table selection
@app.callback(
    Output('map-graph', 'figure'),
    [Input('datatable', 'selected_rows')]
)
def update_map(selected_rows):
    if selected_rows:
        selected_cities = df.iloc[selected_rows]['Station_Name']
        selected_rows = df[df['Station_Name'].isin(selected_cities)]
        fig = px.scatter_geo(selected_rows, lat='Station_Latitude', lon='Station_Longitude', hover_name='Station_Name')
    else:
        fig = px.scatter_geo(df, lat='Station_Latitude', lon='Station_Longitude', hover_name='Station_Name')

    fig.update_geos(projection_type="natural earth")
    return fig


# Define callback to update scatter plot based on selected point or button click
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('map-graph', 'clickData'),
     Input('scatter-button', 'n_clicks')],
    [State('datatable', 'selected_rows')]
)
def update_scatter_plot(clickData, n_clicks, selected_rows):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None

    if triggered_id == 'map-graph.clickData' and clickData:
        selected_point = clickData['points'][0]
        lat = selected_point['lat']
        lon = selected_point['lon']
        selected_row = df[(df['Station_Latitude'] == lat) & (df['Station_Longitude'] == lon)]
        fig = px.scatter(selected_row, x='Station_Latitude', y='Station_Longitude', text='Station_Name')
        fig.update_traces(marker=dict(size=12, color='red'), selector=dict(mode='markers'))
        fig.update_layout(showlegend=False)
        return fig
    elif triggered_id == 'scatter-button.n_clicks' and n_clicks > 0 and selected_rows:
        selected_cities = df.iloc[selected_rows]['Station_Name']
        selected_rows = df[df['Station_Name'].isin(selected_cities)]
        fig = px.scatter(selected_rows, x='Station_Latitude', y='Station_Longitude', text='Station_Name')
        fig.update_layout(showlegend=False)
        return fig
    else:
        return px.scatter()


if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
