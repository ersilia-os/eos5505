import numpy as np
import pandas as pd
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pandas import DataFrame
from numpy import array

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "../predictors"))

from rlm import rlm_gcnn_scaler, rlm_gcnn_model, rlm_gcnn_model_version
from base.gcnn import GcnnBase


class RLMPredictior(GcnnBase):
    """
    Makes RLM stability preditions

    Attributes:
        df (DataFrame): DataFrame containing column with smiles
        smiles_column_index (int): index of column containing smiles
        predictions_df (DataFrame): DataFrame hosting all predictions
    """

    def __init__(self, kekule_smiles: array = None, smiles: array = None):
        """
        Constructor for RLMPredictior class

        Parameters:
            kekule_smiles (Array): numpy array of RDKit molecules
        """

        GcnnBase.__init__(
            self, 
            kekule_smiles, 
            column_dict_key='Predicted Class (Probability)', 
            columns_dict_order = 1, 
            smiles=smiles
        )

        self._columns_dict['Prediction'] = {
            'order': 2,
            'description': 'class label',
            'isSmilesColumn': False
        }

        self.model_name = 'rlm'
        self.model_version = rlm_gcnn_model_version

    def get_predictions(self) -> DataFrame:
        """
        Function that calculates consensus predictions

        Returns:
            Predictions (DataFrame): DataFrame with all predictions
        """

        if len(self.kekule_smiles) > 0:
            start = time.time()
            gcnn_predictions, gcnn_labels = self.gcnn_predict(rlm_gcnn_model, rlm_gcnn_scaler)
            end = time.time()
            print(f'RLM: {end - start} seconds to predict {len(self.predictions_df.index)} molecules')
            self.predictions_df['rlm_proba1'] = pd.Series(np.around(gcnn_predictions, 3))
            proba_df = pd.DataFrame()
            proba_df['rlm_proba1'] = pd.Series(np.around(gcnn_predictions, 3))
        return proba_df

    def get_model_version(self) -> str:
        return self.model_version
