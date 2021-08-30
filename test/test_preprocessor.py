# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:51:38 2021

@author: joshu
"""

import unittest
from obesity.scripts.preprocessors import NCPToCategorical, RareLabelCategoricalEncoder, MultiColumnLabelEncoder
import obesity.scripts.config as config
from obesity.scripts.data_management import load_dataset
import numpy as np


#load test cases for unit test and assign first row with headers as test_input
data = load_dataset(file='test.csv')
test_input = data[0:1]



class TestPreprocessor(unittest.TestCase):
    
    def test_ncp(self):
        
        convert_NCP = NCPToCategorical().fit(test_input).transform(test_input)
        
        self.assertIsInstance(convert_NCP['NCP'].iloc[0], object)
        self.assertEqual(convert_NCP['NCP'].iloc[0], 'Medium')
        
        
    def test_rare_label_encoder(self):
        
       rle = RareLabelCategoricalEncoder(variables=config.RARE_LABEL_ENCODER_VARS)
       result = rle.fit(data).transform(data)
       
       for var in config.RARE_LABEL_ENCODER_VARS:
           unique_vals = result[var].unique()
           self.assertIn('Rare', unique_vals)
       
    def test_multi_column_label_encoder(self):
        
        mcle = MultiColumnLabelEncoder(variables=config.LABEL_ENCODER_VARS)
        result = mcle.fit(data).transform(data)
        
        for var in config.LABEL_ENCODER_VARS:
            self.assertIsInstance(result[var].iloc[0], np.integer) 