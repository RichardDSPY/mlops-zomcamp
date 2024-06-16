from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.homework_03.utils.models.sklearn import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[Series, csr_matrix]]):
    abc = training_set

    return abc