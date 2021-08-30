# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:33:23 2021

@author: joshu
"""

import unittest
import obesity.scripts.predict as predict
from obesity.scripts.data_management import load_dataset


class TestPredict(unittest.TestCase):

    def test_predict(self):
        
        data = load_dataset(file='test.csv')
        
        test_input = data[0:1]
    
        subject = predict.predict(input_data=test_input)
    
        
        self.assertIsNotNone(subject)
        self.assertIsInstance(subject.get('predictions')[0], str)
        self.assertEqual(subject.get('predictions')[0], 'Normal_Weight')
