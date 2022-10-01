import os
import json
import pickle
import logging
from typing import Union, Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config["prod_deployment_path"])
confusion_path = os.path.join(config["output_model_path"])



##############Function for reporting
def score_model(report_data: Union[pd.DataFrame, Type[None]] = None):
    #calculate a confusion matrix using the test data and the deployed model

    if report_data == None:
        report_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    logger.info(type(report_data))
    input_variables = report_data.loc[:, [*set(report_data.columns) - {'exited', 'corporation'}]].copy()
    logger.info(type(input_variables))
    output_variable = report_data['exited'].copy()
    logger.info(type(output_variable))
    predict_target = diagnostics.model_predictions(input_variables)
    logger.info(predict_target)
    confusion = metrics.confusion_matrix(output_variable, predict_target)
    #write the confusion matrix to the workspace
    sns_plot = sns.heatmap(confusion, annot = True, cmap = 'Blues')
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(confusion_path, "confusionmatrix.png"))



if __name__ == '__main__':
    score_model()
