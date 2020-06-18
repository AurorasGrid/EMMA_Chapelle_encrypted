import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# BASE_URL = 'https://127.0.0.1:8282'
BASE_URL = 'https://54.93.177.63:8080'

session = requests.Session()
session.auth = ('user', '$Aurora#5421')

# Get SoC
uri = '/battery/soc'
print(session.get(BASE_URL + uri, verify=False).json())

# Get battery power
uri = '/battery/power'
print(session.get(BASE_URL + uri, verify=False).json())

# Get battery state
uri = '/battery/state'
print(session.get(BASE_URL + uri, verify=False).json())

# Set battery state
uri = '/battery/state'
payload = {'command': 'off'}
print(session.post(BASE_URL + uri, json=payload, verify=False))
