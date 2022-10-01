"""
Module for scoring the model

Author: Bernardo C.
Date: 2022-09-29

"""
import os
import json
import pickle
import logging
from typing import Union, Type
import pandas as pd
from datetime import datetime as dt
from sklearn import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config["output_model_path"]) 

def load_data(input:Union[str, pd.DataFrame, Type[None]] = None
                ) -> pd.DataFrame:
    """
    Function that load the data

    Input:
    ------
        input:(Union[str, pd.DataFrame, Type(None)])
            path, pandas dataframe or the base testdata
    Output:
    ------
        data_frame:(pd.DataFrame)
            loaded dataframe
    """
    logger.info('Initializing load_data function')
    if type(input) == str:
        logger.info(f'Found an path for the data : {str}')
        data_frame = pd.read_csv(input)
    elif type(input) == pd.DataFrame:
        logger.info('Found an path for the data')
        data_frame = input.copy()
    else:
        data_frame = pd.read_csv(os.path.join(test_data_path, 'testdata.csv')) 
    return data_frame

def prepare_data(frame:pd.DataFrame):
    """
    Function to select the columns to train
    
    Input:
    ------
        frame:(pd.DataFrame)
    Output:
    -------
        variable_test: (pd.DataFrame)
            The variable X

        target_test:(pd.DataFrame)
            The target variable y 
    """
    variable_test = frame.loc[:, [*set(frame.columns) - {'exited', 'corporation'}]].copy()
    target_test = frame['exited'].copy()
    return variable_test, target_test


def score_model(model = None,
                input_data: Union[str,
                pd.DataFrame, Type[None]] = None
                ) -> None:
    """
    Function for Module Scoring

    Input:
    ------
        model:(optional)
            Model to take the scores
        input_data:(optional)
            Data to be used for scoring the model

    Output:
    -------
        None
    """
    base_frame = load_data(input = input_data)
    variable_test, target_test = prepare_data(frame = base_frame)

    if model == None:
        with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as file:
            model = pickle.load(file)
       
    predictions = model.predict(variable_test)
    metric_value = metrics.f1_score(target_test, predictions)

    with open(os.path.join(model_path, 'latestscore.txt'), 'a') as score:
        score.write(f"{metric_value} \n")

    return metric_value

if __name__ == "__main__":
    score_model(None, None)
