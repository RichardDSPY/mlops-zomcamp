from typing import Dict, List, Tuple

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def models(*args, **kwargs) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Esta función genera listas de nombres de modelos y metadatos asociados a partir de una cadena de texto 
    separada por comas que se pasa como argumento.

    Args:
        *args: Argumentos posicionales adicionales.
        **kwargs: Argumentos de palabras clave adicionales.
            - models (str): Una cadena de texto separada por comas que contiene nombres completos de clases 
                            de modelos de la biblioteca scikit-learn. Si no se proporciona, 
                            el valor predeterminado es 'linear_model.LinearRegression,linear_model.Lasso'.

    Returns:
        Tuple[List[str], List[Dict[str, str]]]: 
            Una tupla que contiene:
            - Una lista de nombres de modelos (List[str]).
            - Una lista de diccionarios con metadatos, donde cada diccionario contiene el identificador del bloque 
              del modelo correspondiente (List[Dict[str, str]]).

    models: comma separated strings
        linear_model.Lasso
        linear_model.LinearRegression
        svm.LinearSVR
        ensemble.ExtraTreesRegressor
        ensemble.GradientBoostingRegressor
        ensemble.RandomForestRegressor
    """
    model_names: str = kwargs.get(
        'models', 'linear_model.LinearRegression,linear_model.Lasso'
    )
    child_data: List[str] = [
        model_name.strip() for model_name in model_names.split(',')
    ]
    child_metadata: List[Dict] = [
        dict(block_uuid=model_name.split('.')[-1]) for model_name in child_data
    ]

    return child_data, child_metadata
