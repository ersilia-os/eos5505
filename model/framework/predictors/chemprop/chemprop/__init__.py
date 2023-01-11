import sys
import os
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(root, ".."))
print(root)
print(sys.path)

import chemprop.data
import chemprop.features
import chemprop.models
import chemprop.train
import chemprop.web

import chemprop.args
import chemprop.hyperparameter_optimization
import chemprop.interpret
import chemprop.nn_utils
import chemprop.utils
import chemprop.sklearn_predict
import chemprop.sklearn_train

from chemprop._version import __version__
