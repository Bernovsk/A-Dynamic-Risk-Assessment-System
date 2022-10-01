import os
import json
import pickle
import requests
import numpy as np
import pandas as pd
from flask import Flask, session, jsonify, request
import diagnostics
import scoring


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


######################Set up variables for use in our script
app = Flask(__name__)


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    data = request.json.get('dataset_path')
    frame = pd.read_csv(data)
    prediction = diagnostics.model_predictions(frame.drop(['exited', 'corporation'], axis = 1))
    return str(prediction)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    score_value = scoring.score_model()
    #check the score of the deployed model
    return f"F1 score: {score_value}"

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    summary_out = diagnostics.dataframe_summary()
    return summary_out

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostic():        
    execution_time = diagnostics.execution_time()
    count_nulls = diagnostics.count_nulls()
    outdated_packages_list =  diagnostics.outdated_packages_list()
    
    return {'execution_time': execution_time,
            'Nulls Count': count_nulls,
             'outdated_packages': outdated_packages_list}

if __name__ == "__main__":    

    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
