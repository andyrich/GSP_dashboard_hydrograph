from dash import dcc, html

def read_me():
    # Add the README content as a string
    README_CONTENT = html.Div([
        html.H1("Dashboard Documentation"),
        html.P("This web application provides interactive plots and data visualizations based on groundwater and reservoir data. Users can query specific stations, plot types, and other options using query strings in the URL."),
        html.H2("URL Query Parameters"),
        html.Ul([
            html.Li("station_name: Specifies the station to query. Example: station_name=Son0001"),
            html.Li("plot_type: Determines the type of data to plot. Options include:"),
            html.Ul([
                html.Li("MM (default): Manual measurements only"),
                html.Li("Any other value includes pressure data")
            ]),
            html.Li("plot_options: Reserved for future use. Currently NA."),
            html.Li("plot_wet: Indicates whether to include wet season data. Example: plot_wet=true."),
            html.Li("seasonal: Specifies whether to plot seasonal trends. Example: seasonal=true."),
            html.Li("res: If true, displays reservoir storage data."),
            html.Li("pressure_map: If true, displays a pressure map with station locations.")
        ]),
        html.H2("Examples"),
        html.Ul([
            html.Li(html.Code("https://soco-gsp-4a5fb7e8f7c3.herokuapp.com/?station_name=Son0001&plot_type=MM")),
            html.Li(html.Code("https://soco-gsp-4a5fb7e8f7c3.herokuapp.com/?station_name=Son0001&plot_type=PRESS")),
            html.Li(html.Code("https://soco-gsp-4a5fb7e8f7c3.herokuapp.com/?pressure_map=true")),
            html.Li(html.Code("https://soco-gsp-4a5fb7e8f7c3.herokuapp.com/?RES=true")),
            html.Li(html.Code("https://soco-gsp-4a5fb7e8f7c3.herokuapp.com/?RES=true&station_no=MEN"))
        ]),
        html.H2("Features"),
        html.Ul([
            html.Li("Interactive groundwater level plots based on station data."),
            html.Li("Reservoir storage plots for selected stations."),
            html.Li("Pressure map with station locations and metadata.")
        ]),
        # html.H2("Development"),
        # html.Ol([
        #     html.Li("Install the required Python dependencies (e.g., Dash, Plotly)."),
        #     html.Li("Save the script to a file (e.g., app.py)."),
        #     html.Li("Run the script using python app.py."),
        #     html.Li("Access the dashboard at http://127.0.0.1:8050/.")
        # ]),
        html.H2("Notes"),
        html.P("This application is designed for educational and exploratory purposes. Future updates may include additional plot types and customization options.")
    ])

    return  README_CONTENT