import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import urllib.parse

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
        # Parse the URL to extract the 'input_string' parameter
        params = urllib.parse.parse_qs(search[1:])
        input_string = params.get('input_string', ['DefaultString'])[0]
        return input_string
    else:
        return 'DefaultString'

@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('input-string', 'value')]
)
def update_plot(input_string):
    # Create an example Plotly figure (customize as needed)
    import plotly.express as px
    fig = px.scatter(x=[1, 2, 3, 4], y=[1, 2, 3, 4], title=f'Plot for {input_string}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 8053)