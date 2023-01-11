import pickle
import sys
import os

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root, '../chemprop'))
print(root)
print(sys.path)

from chemprop.utils import load_checkpoint, load_scalers

import requests
from io import BytesIO
import tempfile
import shutil
from tqdm import tqdm
import os
from datetime import datetime
from os import path


base_url = 'model/checkpoints'
rlm_base_models_path = 'model/checkpoints'

def load_gcnn_model():
    os.makedirs(f'./model/checkpoints', exist_ok=True)
    print(f'Loading RLM graph convolutional neural network model', file=sys.stdout)
    rlm_gcnn_scaler_path = f'{rlm_base_models_path}/gcnn_model.pt'
    if path.exists(rlm_gcnn_scaler_path):
        rlm_gcnn_scaler, _ = load_scalers(rlm_gcnn_scaler_path)
    else:
        rlm_gcnn_scaler_url = f'{base_url}/gcnn_model.pt'
        rlm_gcnn_scaler_request = requests.get(f'{base_url}/gcnn_model.pt')
        with tqdm.wrapattr(
            open(os.devnull, "wb"),
            "write",
            miniters=1,
            desc=rlm_gcnn_scaler_url.split('/')[-1],
            total=int(rlm_gcnn_scaler_request.headers.get('content-length', 0))
        ) as fout:
            for chunk in rlm_gcnn_scaler_request.iter_content(chunk_size=4096):
                fout.write(chunk)
        with open(rlm_gcnn_scaler_path, 'wb') as rlm_gcnn_scaler_file:
            rlm_gcnn_scaler_file.write(rlm_gcnn_scaler_request.content)
        rlm_gcnn_scaler, _ = load_scalers(rlm_gcnn_scaler_path)

    rlm_gcnn_model = load_checkpoint(rlm_gcnn_scaler_path)

    model_timestamp = datetime.fromtimestamp(os.path.getctime(rlm_gcnn_scaler_path)).strftime('%Y-%m-%d') # get model file creation timestamp
    rlm_gcnn_model_version = 'rlm_' + model_timestamp # generate a model timestamp

    return rlm_gcnn_scaler, rlm_gcnn_model, rlm_gcnn_model_version

rlm_gcnn_scaler, rlm_gcnn_model, rlm_gcnn_model_version = load_gcnn_model()

print(f'Finished loading RLM model files', file=sys.stdout)
