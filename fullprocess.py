import os
import json
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']

##################Check and read new data
def main():
    with open(f"./{output_folder_path}/ingestedfiles.txt", 'r') as file:
        files = {data for data in file.read().split(',') if '.csv' in data}
    
    datadir = {file for file in os.listdir(input_folder_path) if '.csv' in file}

    ##################Deciding whether to proceed, part 1
    if files == datadir:
        return None

    ingestion.merge_multiple_dataframe()

##################Checking for model drift
    with open(f"./{prod_deployment_path}/latestscore.txt", 'r') as file:
        last_f1 = float(file.read())

    model = training.train_model(save = False)
    f1_score = scoring.score_model(model = model, input_data = str(os.path.join(output_folder_path, 'finaldata.csv')))
    if f1_score >= last_f1:
        return None

##################Model Re-train
    training.train_model()

##################Re-deployment
    deployment.store_model_into_pickle()

##################Diagnostics and reporting
    reporting.score_model()
    diagnostics.execution_time()
    diagnostics.dataframe_summary()
    diagnostics.count_nulls()
    diagnostics.outdated_packages_list()
    os.system('python api.py')
    os.system('python apicalls.py')


if __name__ == "__main__":
    main()



