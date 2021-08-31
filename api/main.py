# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 12:53:39 2021

@author: joshu
"""

import sys
if ".." not in sys.path:
    sys.path.insert(0, '..')

import uvicorn
import pandas as pd
from fastapi import FastAPI
import obesity.scripts.predict as ML_predict
from fastapi.encoders import jsonable_encoder
from api.schemas.predict_schema import MultipleObesityInputSchema, PredictionResults


app = FastAPI()

@app.get('/')
async def index():
    return {'message': 'Welcome to my ML API'}

@app.post('/predict', response_model=PredictionResults)
async def predict(input_data: MultipleObesityInputSchema):
    
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    prediction = ML_predict.predict(input_df)
    
    return prediction


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)    