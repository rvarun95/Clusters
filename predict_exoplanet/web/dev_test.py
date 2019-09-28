import requests
import pandas as pd
import numpy as np
from flask import jsonify
import random



url_string = 'http://34.93.86.114:80/'

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

#Reference 
#print(r.text)
#print(r.json())
#print(r.request.body)
#print(r.request.headers)
#print(r.content)
