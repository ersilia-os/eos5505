# Edited returning molecule
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

import numpy as np
import pandas as pd
import sys
import os
import csv
import torch 
from werkzeug.utils import secure_filename
import joblib


# sys.path.insert(0, 'C:/Users/DELL-PC/Desktop/eos/eos5505/model/framework')
sys.path.insert(0, './model/framework')
from predictors.rlm.rlm_predictor import RLMPredictior
from predictors.utilities.utilities import addMolsKekuleSmilesToFrame
from predictors.utilities.utilities import get_similar_mols


global root_route_path
root_route_path = os.getenv('ROOT_ROUTE_PATH', '')

global data_path
data_path = os.getenv('DATA_PATH', '')

if data_path != '' and not os.path.isfile(f'{data_path}predictions.csv'):
    pd.DataFrame(columns=['SMILES', 'model', 'prediction', 'timestamp']).to_csv(f'{data_path}predictions.csv', index=False)

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

# checkpoints directory
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))

input_file = 'model/framework/kekule_smiles.csv'

# read checkpoints (here, simply an integer number: 42)
ckpt = joblib.load(os.path.join(checkpoints_dir, "checkpoints.joblib"))

# model to be run (here, calculate the Molecular Weight and add ckpt (42) to it)
def my_model(smiles_list, ckpt):
    return [MolWt(Chem.MolFromSmiles(smi))+ckpt for smi in smiles_list]


def predict():
    response = {}
    model_error = False
    mol_error = False

    # checking for input - smiles
    # with open(input_file,'r') as smiles:
    #     # smiles = smiles['smiles']
    #     smiles_list = [smile for smile in smiles if smile != '']
    
    # smiles = (pd.read_csv(input_file))
    # smiles = smiles['kekule_smiles'].tolist()
    # smiles_list = [smile for smile in smiles if smile != '']

    with open(input_file, "r") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        smiles_list = [r[1] for r in reader]
    print(smiles_list)
    
    
    if not smiles_list or smiles_list == None:
        mol_error = True

    # checking for input - models
    path = 'model/checkpoints/gcnn_model.pt'
    models = torch.load(path)

    if len(models) == 0 or models == None:
        model_error = True

    # # error handling for invalid inputs
    if mol_error == True and model_error == True:
        response['hasErrors'] = True
        response['errorMessages'] = 'Please choose at least one model and provide at least one input molecule.'
        return response
    elif mol_error == True and model_error == False:
        response['hasErrors'] = True
        response['errorMessages'] = 'Please provide at least one input molecule.'
        return response
    elif mol_error == False and model_error == True:
        response['hasErrors'] = True
        response['errorMessages'] = 'Please choose at least one model.'
        return response

    smi_column_name = 'smiles'
    df = pd.DataFrame([smiles_list], columns=[smi_column_name])
    # print(df)

    try:
        response = predict_df(df, smi_column_name, models)
    except Exception as e:
        response['hasErrors'] = 'Error making a prediction'
        response['errorMessage'] = f'error {e}'
        # response['hasTypeErrors1'] = f'error type: {type(e)}'
        response['haserrors1'] = (e)
        return response
        
    try:
        json_response = response
    except Exception as e:
        response['hasErrors'] = 'Error converting the response to JSON.'
        response['hasTypeErrors2'] = f'error type: {type(e)}'
        response['hasUnknownErrors'] = 'There was an unknown error.'
        response['errorMessage'] = f'error {e}'
        return response
    return json_response


ALLOWED_EXTENSIONS = {'csv', 'txt', 'smi'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_file():
    response = {}

#     # check if the post request has the file part, else throw error message
#     if 'file' not in request.files:
#         response['hasErrors'] = True
#         response['errorMessages'] = 'A file needs to be attached to the request.'
#         return response

    # file = request.files['file']
    file = 'model/framework/kekule_smiles.csv'
    
    # check if the file has a name, else throw error message
    if file.filename == '':
        response['hasErrors'] = True
        response['errorMessages'] = 'A file with a file name needs to be attached to the request.'
        return response

    # check if the file extension is in the allowed list of file extensions (CSV, TXT or SMI)
    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        data = dict(request.form)
        print(data)
        indexIdentifierColumn = int(data['indexIdentifierColumn'])
        models = data['model'].split(';')
        models = [string for string in models if string != '']
        #gcnnOpt = data['gcnnOpt']

        if len(models) == 0 or models == None:
            response['hasErrors'] = True
            response['errorMessages'] = 'Please choose at least one model.'
            return response

        if data['hasHeaderRow'] == 'true': # file has a header row
            df = pd.read_csv(file, header=0, sep=data['columnSeparator'])
        else: # file does not have a header row
            df = pd.read_csv(file, header=None, sep=data['columnSeparator'])
            column_name_mapper = {}
            for column_name in df.columns.values:
                if int(column_name) == indexIdentifierColumn:
                    column_name_mapper[column_name] = 'mol' # check if this is not the case...
                else:
                    column_name_mapper[column_name] = f'col_{column_name}'
            df.rename(columns=column_name_mapper, inplace=True)

        smi_column_name = df.columns.values[indexIdentifierColumn]

        try:
            if len(df.index) > 1000:
                response['hasErrors'] = True
                response['errorMessages'] = 'The input file contains more than 1000 rows which exceeds the limit. Please try again with a maximum of 1000 rows.'
                return response
            else:
                response = predict_df(df, smi_column_name, models)
        except Exception as e:
            response['errorMessage'] = 'Error making a prediction.'
            response['hasTypeError'] = f'error type: {type(e)}'
            response['error'] = (e)
            return(418, 'There was an unknown error.')

        try:
            json_response = response
        except Exception as e:
            response['errorMessage'] = 'Error converting the response to JSON.'
            response['hasResponseError'] = f'response type: {type(response)}'
            response['hasTypeError'] = f'error type: {type(e)}'
            response['error'] = (e)
            return(418, 'There was an unknown error.')

        return json_response
    else:
        response['hasErrors'] = True
        response['errorMessage'] = 'Only csv, txt or smi files can be processed.'
        return response


def predict_df(df, smi_column_name, models):   
    response = {}
    working_df = df.copy()
    
    
    addMolsKekuleSmilesToFrame(working_df, smi_column_name)
    working_df = working_df[~working_df['mols'].isnull() & ~working_df['kekule_smiles'].isnull()]
    if len(working_df.index) == 0:
        response['hasErrors'] = True
        response['errorMessages'] = 'We were not able to parse the smiles you provided'
        return response

    base_models_error_message = 'We were not able to make predictions using the following model(s): '

    # print(len(models))
    for model in models:
        response[model] = {}
        error_messages = []
        
        if model.lower() == 'rlm':
            predictor = RLMPredictior(kekule_smiles = working_df['kekule_smiles'].values, smiles=working_df[smi_column_name].values)
        else:
            break

        pred_df = predictor.get_predictions()

        if data_path != '':
            predictor.record_predictions(f'{data_path}/predictions.csv')
        pred_df = working_df.join(pred_df)
        pred_df.drop(['mols', 'kekule_smiles'], axis=1, inplace=True)

        # columns not present in original df
        diff_cols = pred_df.columns.difference(df.columns)
        df_res = pred_df[diff_cols]

        #response_df = pd.merge(df, pred_df, how='inner', left_on=smi_column_name, right_on=smi_column_name)
        # making sure the response df is of the exact same length (rows) as original df
        response_df = pd.merge(df, df_res, left_index=True, right_index=True, how='inner')

        errors_dict = predictor.get_errors()
        response[model]['hasErrors'] = predictor.has_errors
        model_errors = errors_dict['model_errors']

        if len(model_errors) > 0:
            error_message = base_models_error_message + model_errors.join(', ')
            error_messages.append(error_message)

        response[model]['errorMessages'] = error_messages
        response[model]['columns'] = list(response_df.columns.values)

        columns_dict =  predictor.columns_dict()
        dict_length = len(columns_dict.keys())
        columns_dict[smi_column_name] = { 'order': 0, 'description': 'SMILES', 'isSmilesColumn': True }

        if response_df.shape[0] <= 100: # go for similarity assessment only if the response df contains 100 compounds or less

            # if model.lower() != 'cyp450':
            if model.lower() == 'rlm':
                # for all models except cyp450, calculate the nearest neigbors and add additional column to response_df
                try:
                    sim_vals = get_similar_mols(response_df[smi_column_name].values, model.lower())
                    sim_series = pd.Series(sim_vals).round(2).astype(str)
                    response_df['Tanimoto Similarity'] = sim_series.values
                    columns_dict['Tanimoto Similarity'] = { 'order': 3, 'description': 'similarity towards nearest neighbor in training data', 'isSmilesColumn': False }
                except Exception as e:
                    pass
                    response['errorMessages'] = 'Error calculating similarity'
                    response['hasTypeError'] = f'error type: {type(e)}'
                    response['error'] = (e)
            else:
                try:
                    sim_vals = get_similar_mols(response_df[smi_column_name].values, model.lower())
                    sim_series = pd.Series(sim_vals).round(2).astype(str)
                    response_df['Tanimoto Similarity'] = sim_series.values
                    columns_dict['Tanimoto Similarity'] = { 'order': 7, 'description': 'similarity towards nearest neighbor in training data that was obtained by combining the compounds from all six individual datasets', 'isSmilesColumn': False }
                except Exception as e:
                    response['errorMessages'] = 'Error calculating similarity'
                    response['hasTypeError'] = f'error type: {type(e)}'
                    response['error'] = (e)

        response[model]['mainColumnsDict'] = columns_dict
        response[model]['data'] = response_df.replace(np.nan, '', regex=True).to_dict(orient='records')
        
        if model.lower() in ['rlm']:
            response[model]['model_version'] = predictor.get_model_version()
        return response
    return response

# outputs = my_model(smiles_list, ckpt)

# # write output in a .csv file
# with open(output_file, "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["value"]) # header
#     for o in outputs:
#         writer.writerow([o])
        
x = predict()
print(x)
