import pandas as pd
import numpy as np


def load_excel(path, **kwargs):
    """
    Load excel with the specified kwargs
    """
    return pd.read_excel(path, **kwargs)


def load_csv(path, **kwargs):
    """
    Load csv with the specified kwargs
    """
    return pd.read_csv(path, **kwargs)


def store_labels(df, path, **kwargs):
    """
    Store the labels file
    """
    df.to_csv(path, **kwargs)


def store_features(path, video_features):
    """
    Store the features file
    """
    np.save(path, video_features)