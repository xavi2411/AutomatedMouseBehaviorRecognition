import os
import pandas as pd
import numpy as np
import yaml
import json


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


def load_dataset(path, split, sets=['train', 'test']):
    """
    Load the dataset
    """
    dataset = {}
    for set in sets:
        feature_files = [
            f for f in sorted(os.listdir(os.path.join(path, 'features')))
            if f.split('.')[0] in split[set]]
        label_files = [
            f for f in sorted(os.listdir(os.path.join(path, 'labels')))
            if f.split('.')[0] in split[set]]
        features = []
        labels = []
        for f, l in zip(feature_files, label_files):
            features.append(np.load(os.path.join(path, 'features', f)))
            labels.append(pd.read_csv(os.path.join(path, 'labels', l)).values)
        dataset[set] = {
            'features': np.concatenate(features, axis=0),
            'labels': np.concatenate(labels, axis=0),
            'num_videos': len(feature_files),
        }
    return dataset


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


def store_output(objects, path, exec_name):
    """
    Store the output objects
    """
    path = os.path.join(path, exec_name)
    os.mkdir(path)
    for key, obj in objects.items():
        if key == 'settings':
            file = os.path.join(path, 'settings.yaml')
            with open(file, 'w') as f:
                yaml.dump(obj, f)
        if key == 'best_params':
            file = os.path.join(path, 'best_params.yaml')
            with open(file, 'w') as f:
                yaml.dump(obj, f, default_flow_style=False)