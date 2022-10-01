import os
import sys
import json
import pickle
import timeit
import logging
import subprocess
from typing import Union, Type
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

##################Load config.json and get environment variables

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config["prod_deployment_path"])

def read_data():
    return pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

##################Function to get model predictions
def model_predictions(data : Union[pd.DataFrame, Type[None]] = None):
    logger.info('Verifying if is necessary load the test data')
    if type(data) == type(None):
        logger.info('Reading Test data')
        data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
        data = data.drop(['exited', 'corporation'], axis = 1)
    #read the deployed model and a test dataset, calculate predictions
    logger.info('Loading the model')
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    logger.info('The model has been loaded')
    prediction = model.predict(data)
    logger.info('The prediction is over')
    return prediction

##################Function to get summary statistics
def dataframe_summary(data: Union[pd.DataFrame, Type[None]] = None):
    if data == None:
        data = read_data()

    statistic = data.describe(
        include=np.number).loc[['mean', 'std', '50%']].T.rename(
            columns = {'50%':'Median'}).T.to_dict()

    return statistic

def count_nulls(data_frame: Union[pd.DataFrame, Type[None]] = None):

    if data_frame == None:
        data_frame = read_data()
    null_count = data_frame.isna().sum(axis = 0)
    null_pct = (100*null_count/data_frame.shape[0]).to_dict()
    return null_pct

##################Function to get timings
def execution_time():
    startingestion = timeit.default_timer()
    os.system('python ingestion.py')
    starttraining = timeit.default_timer()
    os.system('python training.py')
    end_processes = timeit.default_timer()
    return [starttraining - startingestion, end_processes - starttraining]

##################Function to check dependencies
def outdated_packages_list():
    package_info = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(package_info).split('wheel\n')[1:]


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
