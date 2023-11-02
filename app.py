import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import urllib.parse
import wiski_data
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Graph(id='interactive-plot')
])

@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('url', 'search')]
)
def update_inputs_from_url(search):
    if search:
        print(search)
        params = urllib.parse.parse_qs(search[1:])
        station_name = params.get('station_name', ['DefaultStation'])[0]
        plot_type = params.get('plot_type', ['MM'])[0]
        plot_options = params.get('plot_options', [''])[0]

        if plot_type == 'MM':
            # fig = px.scatter(data, x='x', y='y', title=f'Scatter Plot for {station_name}',
            #                  **parse_plot_options(plot_options))
            print("UNDER MANUAL MEASUREMENT\n\n\n")
            x = wiski_data.wiski_plot(station_name)
            x.get_station_pars(remove_pt=True)
            fig = x.plot_gw()
        elif plot_type == 'PRESS':
            print("UNDER PRESSURE\n\n\n")
            x = wiski_data.wiski_plot(station_name)
            x.get_station_pars(remove_pt=False)
            fig = x.plot_gw()
        else:
            fig = px.scatter(data, x='x', y='y', title=f'Scatter Plot for {station_name}',
                             **parse_plot_options(plot_options))

        return fig
    else:
        print("serach is NONE")
        return {}


# @app.callback(
#     Output('interactive-plot', 'figure'),
#     [Input('station-name', 'value'),
#      Input('plot-type', 'value'),
#      Input('plot-options', 'value')]
# )
# def update_plot(station_name, plot_type, plot_options):
#     # Create an example Plotly figure based on inputs (customize as needed)
#     import plotly.express as px
#
#     data = {'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]}
#
#     if plot_type == 'MM':
#         # fig = px.scatter(data, x='x', y='y', title=f'Scatter Plot for {station_name}',
#         #                  **parse_plot_options(plot_options))
#         print("UNDER MANUAL MEASUREMENT\n\n\n")
#         x = wiski_data.wiski_plot(station_name)
#         x.get_station_pars(remove_pt=True)
#         fig = x.plot_gw()
#     elif plot_type == 'PRESS':
#         print("UNDER PRESSURE\n\n\n")
#         x = wiski_data.wiski_plot(station_name)
#         x.get_station_pars(remove_pt=False)
#         fig = x.plot_gw()
#     else:
#         fig = px.scatter(data, x='x', y='y', title=f'Scatter Plot for {station_name}',
#                          **parse_plot_options(plot_options))
#
#     return fig


def parse_plot_options(options):
    parsed_options = {}
    if options:
        options = options.split(',')
        for option in options:
            key, value = option.split('=')
            parsed_options[key.strip()] = value.strip()
    return parsed_options


if __name__ == '__main__':
    app.run_server(debug=True, port = 8054)