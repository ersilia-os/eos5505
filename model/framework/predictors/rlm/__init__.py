import sys
import os

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

from datetime import datetime
from predictors.chemprop.chemprop.utils import load_checkpoint, load_scalers


base_url = os.path.abspath(os.path.join(root, '../../../checkpoints'))
rlm_base_models_path = os.path.abspath(os.path.join(root, '../../../checkpoints'))

def load_gcnn_model():
    print(f'Loading RLM graph convolutional neural network model', file=sys.stdout)
    rlm_gcnn_scaler_path = f'{rlm_base_models_path}/gcnn_model.pt'
    rlm_gcnn_scaler, _ = load_scalers(rlm_gcnn_scaler_path)
    rlm_gcnn_model = load_checkpoint(rlm_gcnn_scaler_path)

    # get model file creation timestamp
    model_timestamp = datetime.fromtimestamp(os.path.getctime(rlm_gcnn_scaler_path)).strftime('%Y-%m-%d')
    # generate a model timestamp
    rlm_gcnn_model_version = 'rlm_' + model_timestamp 

    return rlm_gcnn_scaler, rlm_gcnn_model, rlm_gcnn_model_version

rlm_gcnn_scaler, rlm_gcnn_model, rlm_gcnn_model_version = load_gcnn_model()

print(f'Finished loading RLM model files', file=sys.stdout)
