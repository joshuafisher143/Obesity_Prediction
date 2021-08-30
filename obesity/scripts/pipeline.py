# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:12:03 2021

@author: joshu
"""


from obesity.scripts.preprocessors import NCPToCategorical, MultiColumnLabelEncoder, RareLabelCategoricalEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import obesity.scripts.config as config
from sklearn.compose import ColumnTransformer


pipe = Pipeline(
    [
        (
            "ncp_categorizer",
            NCPToCategorical(),
        ),
        (
            "label_encoder",
            MultiColumnLabelEncoder(variables=config.LABEL_ENCODER_VARS),
        ),
        (
            'rare_label_encder',
            RareLabelCategoricalEncoder(variables=config.RARE_LABEL_ENCODER_VARS),
        ),
        (
            'encoder_transformer', ColumnTransformer(transformers=[("one_hot_encoder",
            OneHotEncoder(), config.DUMMY_VARIABLE_VARS)], remainder='passthrough'),
        ),
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel='linear', random_state=0))
    ]
)