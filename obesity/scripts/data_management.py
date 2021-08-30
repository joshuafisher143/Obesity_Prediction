# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:57:43 2021

@author: joshu
"""

import pandas as pd
import obesity.scripts.config as config
import joblib

#load dataset
def load_dataset(file):
    data = pd.read_csv(f"{config.data_dir}/{file}")
    return data
    
    
#save model
def save_model(pipeline):    
    save_file_name = f"{config.PIPELINE_NAME}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    # remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline, save_path)
    
    
#load saved model
def load_model(file):

    file_path = config.TRAINED_MODEL_DIR / file
    trained_model = joblib.load(filename=file_path)
    return trained_model