import os
import json
import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"

with open('config.json','r') as f:
    config = json.load(f) 
model_path = os.path.join(config['output_model_path'])


#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}:8000/prediction', json={'dataset_path': 'testdata/testdata.csv'}).text
response2 = requests.get(f'{URL}:8000/scoring').text
response3 = requests.get(f'{URL}:8000/summarystats').text
response4 = requests.get(f'{URL}:8000/diagnostics').text

#combine all API responses
responses = "%s \n %s \n %s \n %s" % (response1, response2, response3, response4)

#write the responses to your workspace
with open(os.path.join(model_path, 'apireturns.txt'), 'w') as file:
    file.write(responses)


