# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:52:12 2021

@author: joshu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin



class NCPToCategorical(BaseEstimator, TransformerMixin):
    """
    Bin column "NCP" from float to three categories: Low, Medium, High
    Low: NCP <= 2    
    Medium: 2 < NCP <= 3
    High: NCP > 3
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['NCP'] = np.where(X['NCP'] >3, 'High', np.where(X['NCP'] <=2, 'Low', 'Medium'))
        
        return X

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    #Bin numeric variable into categories
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.variables:
            X[col] = LabelEncoder().fit_transform(X[col])
        return X
    
class GetDummies(BaseEstimator, TransformerMixin):
    #Convert categorical variables with more than 2 categories into dummy variables
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X = pd.get_dummies(X, columns=self.variables)

        return X
    
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare"
            )

        return X













        