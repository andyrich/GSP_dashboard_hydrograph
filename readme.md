# GSP map query

## how to run:

1) make a virtual env like: 

    python -m venv venv


2) activate the environment: 

    '''venv\Scripts\activate''' for windows
    
    source venv/bin/activate' for linux
    
3) install the requirements

    pip install --r requirements.txt

4) start the application:

    python app_multi.py (for the api-like version)
    python app_time_since_update.py (for the map showing time since updates for manual measurements or pressure transducers)
    python app_map.py (map of hydrographs with selection for depth/rmp's etc)

### to test app, try:

    http://127.0.0.1:8053/?station_name=Son0001

     