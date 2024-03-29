# imports
import os
import csv
import sys
import pandas as pd
import numpy as np
from rdkit import Chem

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

from predictors.rlm.rlm_predictor import RLMPredictior
from predictors.utilities.utilities import addMolsKekuleSmilesToFrame

#Arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

#Model
def my_model(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    kek_mols = []
    for mol in mols:
        if mol is not None:
            Chem.Kekulize(mol)
        kek_mols += [mol]
    kek_smiles = [Chem.MolToSmiles(mol,kekuleSmiles=True) for mol in kek_mols]
    predictor = RLMPredictior(kekule_smiles = np.asarray(kek_smiles), smiles = np.asarray(smiles_list))
    pred_df = predictor.get_predictions()
    
    return pred_df


# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    smiles_list = [r[0] for r in reader]
    
# run model
output = my_model(smiles_list)
     
# write output in a .csv file
output.to_csv(output_file, index=False)