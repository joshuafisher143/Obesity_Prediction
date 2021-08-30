# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:35:37 2021

@author: joshu
"""


import obesity.scripts.config as config
from obesity.scripts.data_management import load_dataset
from obesity.scripts.data_management import save_model
import obesity.scripts.pipeline as pipeline
from sklearn.model_selection import train_test_split


def train_pipeline():
    """Train the pipeline"""
    
    #load data
    data = load_dataset(config.train_file)
    
    #train model
    pipeline.pipe.fit(data[config.FEATURES], data[config.TARGET])
    
    #save model
    save_model(pipeline.pipe)