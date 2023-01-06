import sys
sys.path.insert(0, './model/framework/predictors')

import chemprop.chemprop.data
import chemprop.chemprop.features
import chemprop.chemprop.models
import chemprop.chemprop.train
import chemprop.chemprop.web

import chemprop.chemprop.args
import chemprop.chemprop.hyperparameter_optimization
import chemprop.chemprop.interpret
import chemprop.chemprop.nn_utils
import chemprop.chemprop.utils
import chemprop.chemprop.sklearn_predict
import chemprop.chemprop.sklearn_train

from chemprop._version import __version__
