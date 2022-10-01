
import os
import json
import shutil


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 



def store_model_into_pickle():
    shutil.copyfile(os.path.join(dataset_csv_path, "ingestedfiles.txt"),
                    os.path.join(prod_deployment_path, "ingestedfiles.txt"))

    shutil.copyfile(os.path.join(model_path, 'trainedmodel.pkl'),
                    os.path.join(prod_deployment_path, 'trainedmodel.pkl'))

    shutil.copyfile(os.path.join(model_path, 'latestscore.txt'),
                    os.path.join(prod_deployment_path, 'latestscore.txt'))
   
        
if __name__ == "__main__":
    store_model_into_pickle()
        

