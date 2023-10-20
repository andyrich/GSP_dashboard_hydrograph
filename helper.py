import os
import requests
from kiwis_pie import KIWIS

def wiski_request_ssl(url):
    '''
    request data from wiski using only pandas
    :param url:
    :return: content to be read with pd.read_csv/read_html etc

    Examples:
    url = www.example.data.com
    content = wiski_request_ssl(url)
    df = pd.read_csv(content)

    '''

    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "kisters-net-chain.pem")
    content = requests.get(url, verify =DATA_PATH).content

    return content

def get_kiwis():
    '''
    get kiwis object
    Returns:

    '''

    kurl = 'https://www2.kisters.net/sonomacountygroundwater/KiWIS/KiWIS?'

    #add the certif file bc kiwis ssl file is problematic
    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "kisters-net-chain.pem")

    k = KIWIS(kurl, verify_ssl = DATA_PATH)

    return k