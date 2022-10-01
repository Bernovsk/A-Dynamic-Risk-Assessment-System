import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def data_ingestion_track(file_name : str,
                        ) -> None:
    """
    Registration of the data that was used to create the finaldata.csv
    during the ingestion step

    Input:
    ------
        file_name: (str)
            Name of the file to be stored
        key: (str)
            Order of the in the read step

    Output:
    -------
        (None)
    """

    with open(f"./{output_folder_path}/ingestedfiles.txt", 'a') as file:
        file.write(f"{file_name}")
        file.write(',')

# Function for data ingestion
def merge_multiple_dataframe():
    """
    Module that compile multiple dataframes into only one

    Input:
    ------
        (None)

    Output:
    ------
        (None)
    """
    # check for datasets, compile them together, and write to an output file
    get_data_files = [file for file in os.listdir(
        os.path.join(
            '.',
            input_folder_path)) if '.csv' in file]
    
    for key, file in enumerate(get_data_files):
        if key == 0:
            main_frame = pd.read_csv(os.path.join(input_folder_path,file))
        else:
            append_frames = pd.read_csv(os.path.join(input_folder_path,file))
            main_frame = pd.concat([main_frame, append_frames])
        data_ingestion_track(file_name= file)

    main_frame = main_frame.drop_duplicates()
    main_frame.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index = False)

if __name__ == '__main__':
    merge_multiple_dataframe()
