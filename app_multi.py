import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import urllib.parse
import wiski_data
import wiski_census
import plotly.express as px
from dash import dash_table
import reservoir_storage

info = None

'''
query site with the following:
https://soco-gsp-4a5fb7e8f7c3.herokuapp.com/?station_name=Son0001&plot_type=PRESS

query strings like:
        station_name
        plot_type MM = manual measurements only (default), anything else will have pressure data
        plot_options NA
        plot_wet true/false
        seasonal ie true/false

'''
app = dash.Dash(__name__)

server = app.server

# Create the layout of the app
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Graph(id="interactive-plot"),
    html.Div(
        id="datatable-container",
        children=[],  # Initially empty datatable container
        # style={"display": "none"},  # Initially hidden
    ),
])


# Callback function to update plot or datatable based on URL parameters
@app.callback(
    Output("interactive-plot", "figure"),
    Output("datatable-container", "children"),
    [Input("url", "search")],
    cache_reasoning="no-action"  # Prevent unnecessary cache invalidation
)
def update_inputs_from_url(search):
    global info  # Access the global variable

    if search:
        print(search)
        params = urllib.parse.parse_qs(search[1:])
        station_name = params.get('station_name', ['DefaultStation'])[0]
        plot_type = params.get('plot_type', ['MM'])[0]
        plot_options = params.get('plot_options', [''])[0]
        plot_wet = params.get('plot_wet', ['True'])[0].lower() == 'true'
        seasonal = params.get('seasonal', ['True'])[0].lower() == 'true'
        RMP = params.get('rmp', ['False'])[0].lower() == 'true'
        ISW = params.get('isw', ['False'])[0].lower() == 'true'

        PRESSURE_MAP = params.get('PRESSURE_MAP', ['False'])[0].lower() == 'true'

        ressy = params.get('RES',['False'])[0].lower() == 'true'

        if PRESSURE_MAP:
            if info is None:
                print('loading PT')
                info = wiski_census.get_all()
                info = info.astype({"station_longitude":float,
                                    "station_latitude":float})
                print('done loading PT...\n')

            # Datatable creation (if data exists)
            dtable = dash_table.DataTable(
                data=info.to_dict("records"),
                columns=[{"name": col, "id": col} for col in info.columns],
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=True,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
            ),

            fig = px.scatter_geo(info, lon = 'station_longitude', lat = 'station_latitude',
                                 # locations="station_name",
                                 color="Elapsed Time",
                                     hover_name="station_name",
                                     size="Elapsed Time",
                                 fitbounds='locations'
                                 # # animation_frame="Elapsed Time",
                                 # projection="natural earth"
                                 )
            fig.update_layout(mapbox_style="open-street-map")

            # fig.show()
            return fig, dtable  # Return figure and datatable

        elif ressy:
            if station_name.lower() == "son":
                x, stor = reservoir_storage.get_son()
                fig = reservoir_storage.plot_son(x.reset_index(), stor)
            else:
                act_mendo, stor_mendo = reservoir_storage.get_men()
                act_mendo = act_mendo.reset_index()
                stor_mendo = stor_mendo.reset_index()
                fig = reservoir_storage.plot_men(act_mendo, stor_mendo)


            return fig, {}
        else:
            remove_pt = plot_type =='MM'

            x = wiski_data.wiski_plot(station_name)
            x.get_station_pars(remove_pt=remove_pt)
            fig = x.plot_gw(plot_wet = plot_wet, seasonal=seasonal, RMP=RMP, isw = ISW)


            return fig, {}
    else:
        print("serach is NONE")
        # If no search parameters, return empty figure/datatable
        return {}, None


def parse_plot_options(options):
    parsed_options = {}
    if options:
        options = options.split(',')
        for option in options:
            key, value = option.split('=')
            parsed_options[key.strip()] = value.strip()
    return parsed_options


if __name__ == '__main__':
    app.run_server(debug = True)