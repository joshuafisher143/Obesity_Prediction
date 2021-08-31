# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:47:52 2021

@author: joshu
"""


from pathlib import Path
import obesity

"""
Root Directory
"""
ROOT = Path(obesity.__file__).resolve().parent
TRAINED_MODEL_DIR = ROOT / "trained_model"


"""
Pipeline save file name
"""

PIPELINE_NAME = "svc_classification"


"""
Directory and files to training and test data
"""

data_dir = ROOT / "data/train_test"
train_file = 'train.csv'
test_file = 'test.csv'


"""
Below are the variables that are preprocessed in the pipeline
"""

LABEL_ENCODER_VARS = ['Gender',
                      'family_history_with_overweight', 
                      'FAVC',
                      'SMOKE',
                      'SCC']

RARE_LABEL_ENCODER_VARS = ['CAEC', 
                           'CALC', 
                           'MTRANS']

DUMMY_VARIABLE_VARS = ['NCP', 
                       'CAEC', 
                       'CALC', 
                       'MTRANS']



"""
Training and target features
"""

FEATURES = ['Gender',
            'Age',
            'Height',
            'Weight',
            'family_history_with_overweight',
            'FAVC',
            'FCVC',
            'NCP',
            'CAEC',
            'SMOKE',
            'CH2O',
            'SCC',
            'FAF',
            'TUE',
            'CALC',
            'MTRANS']

TARGET = 'NObeyesdad'


