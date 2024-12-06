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
    does not work -- python app_time_since_update.py (for the map showing time since updates for manual measurements or pressure transducers)
    python app_gwl_hydros.py (map of hydrographs with selection for depth/rmp's etc)


# to push to heroku
make sure you're on the correct web-app using:  
  

    heroku git:remote -a soco-gsp

or  

    heroku git:remote -a gwmonnet  

## set procfile to correct webapp
make sure procfile refers to which webapp you're pusing to.

for socogsp, file is app_multi and for monitoring network it is app_gwl_hydros

## push to heroku

`git push heroku master`  


### to test app, try:

    http://127.0.0.1:8053/?station_name=Son0001

     