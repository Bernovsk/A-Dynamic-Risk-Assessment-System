"""
Module for training the model

Author: Bernardo C.
Date: 2022-09-29
"""

import os
import json
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model(save : bool = True):
    """
    Function to train an initial model
    """
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    variable_data = data.loc[:, [*set(data.columns) - {'exited', 'corporation'}]].copy()

    target_data = data['exited'].copy()
    
    #use this logistic regression for training
    classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model = classifier.fit(variable_data, target_data)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    if save:
        pickle.dump(model, open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb'))
        
    return model

if __name__ == "__main__":
    train_model()