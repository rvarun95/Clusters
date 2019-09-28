import requests
import pandas as pd
import numpy as np
from flask import jsonify
import random


#Application Base URL
url_string = 'http://34.93.86.114:80/'
#url_string = 'http://0.0.0.0:5205/'

#Request1 to do GET on home url
r = requests.get(url_string)
print(r.text)

#Request2 to do GET on hithere url
url = url_string + 'hithere'
r = requests.get(url)
print(r.text)


#Request 3 and 4 to do POST on add url
url = url_string + 'add'
requestJson1 = {"x":100, "y":20}
r = requests.post(url,json=requestJson1)
print(r.json())
requestJson2 = {"x":100, "y":200}
r = requests.post(url,json=requestJson2)
print(r.json())

#Request to do exoplanet_predict
#Load full data for test
df = pd.read_csv('exoTest.csv')
#Load sample data for development
#df = pd.read_csv('few_rows_few_cols_exoTest.csv')
#df = pd.read_csv('all_rows_few_cols_exoTest.csv')
#df = pd.read_csv('few_rows_all_cols_exoTest.csv')
url = url_string + 'exoplanet_predict'
for test_num in range(10):
    sample = random.randrange(1,370)
    #sample = test_num
    request_arr = np.array(df.iloc[sample])
    request_json = df.iloc[sample].to_json(orient='records')
    requestJson = {
        "request_array" : request_arr.tolist(),
        "request_json" : request_json
    }
    response = requests.post(url,json=requestJson)
    print("Row selected for testing: %03d " % (sample), response, response.text)

sample = random.randrange(1,5)
request_arr = np.array(df.iloc[sample])
request_json = df.iloc[sample].to_json(orient='records')
requestJson = {
    "request_array" : request_arr.tolist(),
    "request_json" : request_json
}
response = requests.post(url,json=requestJson)
print("Row selected for testing: %03d " % (sample), response, response.text)


#Reference
#print(response.text)
#print(response.json())
#print(response.request.body)
#print(response.request.headers)
#print(response.content)
