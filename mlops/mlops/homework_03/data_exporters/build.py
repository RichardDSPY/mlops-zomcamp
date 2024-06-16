from typing import List, Tuple

from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.homework_03.utils.encoders import vectorize_features
from mlops.homework_03.utils.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs
) -> Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    BaseEstimator,
]:
    """
    Exporta matrices dispersas y series objetivo para entrenamiento y validación.

    Args:
        data (tuple): Una tupla de tres DataFrames (df, df_train, df_val) que contiene
                      todos los datos, datos de entrenamiento y datos de validación, respectivamente.
        *args: Argumentos adicionales.
        **kwargs: Argumentos clave adicionales, especialmente:
                    - target (str): Nombre de la columna objetivo a predecir. Por defecto es 'duration'.

    Returns:
        tuple: Una tupla que contiene:
               - X (csr_matrix): Matriz dispersa con características de todos los datos.
               - X_train (csr_matrix): Matriz dispersa con características de los datos de entrenamiento.
               - X_val (csr_matrix): Matriz dispersa con características de los datos de validación.
               - y (Series): Serie con la variable objetivo de todos los datos.
               - y_train (Series): Serie con la variable objetivo de los datos de entrenamiento.
               - y_val (Series): Serie con la variable objetivo de los datos de validación.
               - dv (BaseEstimator): Estimador ajustado para las características.
    """

    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    # Con toda la data, pero para que?
    X, _, _ = vectorize_features(select_features(df))
    y: Series = df[target]

    X_train, X_val, dv = vectorize_features(
        select_features(df_train),
        select_features(df_val),
    )
    y_train = df_train[target]
    y_val = df_val[target]

    return X, X_train, X_val, y, y_train, y_val, dv

@test
def test_dataset(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        len(y.index) == X.shape[0]
    ), f'Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}'