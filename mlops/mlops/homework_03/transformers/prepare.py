import pandas as pd

from mlops.homework_03.utils.cleaning import clean
from mlops.homework_03.utils.feature_selector import select_features
from mlops.homework_03.utils.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:

    split_on_feature = kwargs.get('split_on_feature', 'tpep_pickup_datetime')
    split_on_feature_value = kwargs.get('split_on_feature_value', '2023-03-20')
    target = kwargs.get('target', 'duration')

    df = clean(df)
    df = select_features(df, features=[split_on_feature, target])

    df_train, df_val = split_on_value(
        df,
        split_on_feature,
        split_on_feature_value
    )

    return df, df_train, df_val


