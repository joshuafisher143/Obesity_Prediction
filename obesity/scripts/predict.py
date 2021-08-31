# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:41:11 2021

@author: joshu
"""

from obesity.scripts.data_management import load_model
import pandas as pd
import obesity.scripts.config as config


svc_pipeline = load_model('svc_classification.pkl')

def predict(input_data):
    
    data = pd.DataFrame(input_data, columns=config.FEATURES)
    
    prediction = svc_pipeline.predict(data)
    
    results = {'predictions': list(prediction)}
    
    return results