# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:09:14 2021

@author: joshu
"""

from typing import List, Optional
from pydantic import BaseModel


class PredictionResults(BaseModel):
    predictions: Optional[List[str]]


class ObesityInputSchema(BaseModel):
    Gender: Optional[str]
    Age: Optional[int]
    Height: Optional[float]
    Weight: Optional[int]
    family_history_with_overweight: Optional[str]
    FAVC: Optional[str]
    FCVC: Optional[int]
    NCP: Optional[int]
    CAEC: Optional[str]
    SMOKE: Optional[str]
    CH2O: Optional[int]
    SCC: Optional[str]
    FAF: Optional[int]
    TUE: Optional[int]
    CALC: Optional[str]
    MTRANS: Optional[str]
    
class MultipleObesityInputSchema(BaseModel):
    inputs: List[ObesityInputSchema]