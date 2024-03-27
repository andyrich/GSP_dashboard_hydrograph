import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import dash_table
import wiski_data
import pressure_transducers

# Sample DataFrame
df = pressure_transducers.get_all()
df = df.astype({"station_longitude": float,
                "station_latitude": float})

df = df.drop(columns=['ts_name', 'from', 'to'])

# Rename latitude and longitude columns
df.rename(columns={'station_latitude': 'Station_Latitude', 'station_longitude': 'Station_Longitude',
                   'station_name': "Station_Name", "Elapsed Time": "Elapsed_Time"}, inplace=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    dcc.Graph(id='map-graph'),
    html.Button('Generate Scatter Plot', id='scatter-button', n_clicks=0),
    dcc.Graph(id='scatter-plot'),
    dcc.RadioItems(
        id='filter-radio',
        options=[
            {'label': 'Manual Measurements', 'value': 'Manual'},
            {'label': 'Pressure Transducers', 'value': 'Pressure'}
        ],
        value='Manual',
        labelStyle={'display': 'inline-block'}
    ),
    dash_table.DataTable(
        id='datatable',
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.to_dict('records'),
        row_selectable='multi',
        filter_action='native',
        sort_action='native',
        style_table={'overflowX': 'auto'}
    )
])


# Define callback to update DataFrame based on radio button selection
@app.callback(
    Output('datatable', 'data'),
    [Input('filter-radio', 'value')]
)
def update_dataframe(value):
    # Implement your filtering condition here
    # Example: filtered_df = df[df['some_column'] > some_value]

    filtered_df = df[df['Param'] == value]
    # Replace the above line with your filtering condition
    # filtered_df = df  # Placeholder, replace with your actual filtering logic
    return filtered_df.to_dict('records')



# Define callback to update map based on table selection
@app.callback(
    Output('map-graph', 'figure'),
    [Input('datatable', 'selected_rows'),
     Input('filter-radio', 'value')]
)
def update_map(selected_rows, press):
    print(press)
    # No change in this callback function
    if selected_rows:
        selected_cities = df.iloc[selected_rows]['Station_Name']
        selected_rows = df[df['Station_Name'].isin(selected_cities)]
        print(selected_rows)
        selected_rows = selected_rows.query(f"Param=='{press}'")
        fig = px.scatter_mapbox(selected_rows, lon='Station_Longitude', lat='Station_Latitude',
                                color="Number of Months Since Last Measurement",
                                hover_name="Station_Name",
                                hover_data=["Station_Name", "station_no", "Number of Months Since Last Measurement"],
                                size="Number of Months Since Last Measurement")
    else:
        print(df.query(f"Param=='{press}'"))
        fig = px.scatter_mapbox(df.query(f"Param=='{press}'"), lon='Station_Longitude', lat='Station_Latitude',
                                color="Number of Months Since Last Measurement",
                                hover_name="Station_Name",
                                hover_data=["Station_Name", "station_no", "Number of Months Since Last Measurement"],
                                size="Number of Months Since Last Measurement")
    fig.update_layout(mapbox_style="open-street-map")
    return fig


# Define callback to update scatter plot based on selected point or button click
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('map-graph', 'clickData'),
     Input('scatter-button', 'n_clicks')],
    [State('datatable', 'selected_rows')]
)
def update_scatter_plot(clickData, n_clicks, selected_rows):
    # No change in this callback function
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None

    if triggered_id == 'map-graph.clickData' and clickData:
        selected_point = clickData['points'][0]['hovertext']
        x = wiski_data.wiski_plot(selected_point)
        x.get_station_pars(remove_pt=False)
        fig = x.plot_gw(plot_wet=False, seasonal=False, RMP=False, isw=False)
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
