import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import urllib.parse
import wiski_data
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Input(id='input-string', type='text', placeholder='Enter a string'),
    dcc.Graph(id='interactive-plot')
])

@app.callback(
    Output('input-string', 'value'),
    [Input('url', 'search')]
)
def update_input_from_url(search):
    if search:
        # Parse the URL to extract the 'station_name' parameter
        params = urllib.parse.parse_qs(search[1:])
        input_string = params.get('station_name', ['DefaultString'])[0]
        return input_string
    else:
        return 'Son0001'

@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('input-string', 'value')]
)
def update_plot(input_string):
    # Create an example Plotly figure (customize as needed)
    # import plotly.express as px
    # fig = px.scatter(x=[1, 2, 3, 4], y=[1, 2, 3, 4], title=f'Plot for {input_string}')
    x =wiski_data.wiski_plot(input_string)
    x.get_station_pars(remove_pt=True)
    fig = x.plot_gw()

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 8053)