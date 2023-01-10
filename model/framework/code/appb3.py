import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

sys.path.insert(0, './model/framework')

from predictors.rlm.rlm_predictor import RLMPredictior
from predictors.utilities.utilities import addMolsKekuleSmilesToFrame


def predict_df(df, smi_column_name='smiles', models=['rlm']):

    #interpret = False
    #if gcnnOpt == 'yes':
    #    interpret = True

    response = {}
    working_df = df.copy()
    print(working_df)
    addMolsKekuleSmilesToFrame(working_df, smi_column_name)
    working_df = working_df[~working_df['mols'].isnull() & ~working_df['kekule_smiles'].isnull()]
    print(working_df)

    base_models_error_message = 'We were not able to make predictions using the following model(s): '

    for model in models:
        response[model] = {}
        error_messages = []
        
        if model.lower() == 'rlm':
            predictor = RLMPredictior(kekule_smiles = working_df['kekule_smiles'].values, smiles=working_df[smi_column_name].values)
            print(predictor)
        else:
            break

        pred_df = predictor.get_predictions()
        print(pred_df)

        pred_df = working_df.join(pred_df)
        print(pred_df)
        pred_df.drop(['mols', 'kekule_smiles'], axis=1, inplace=True)

        # columns not present in original df
        diff_cols = pred_df.columns.difference(df.columns)
        print(diff_cols)
        df_res = pred_df[diff_cols]
        print(df_res)

        #response_df = pd.merge(df, pred_df, how='inner', left_on=smi_column_name, right_on=smi_column_name)
        # making sure the response df is of the exact same length (rows) as original df
        response_df = pd.merge(df, df_res, left_index=True, right_index=True, how='inner')

        return response_df