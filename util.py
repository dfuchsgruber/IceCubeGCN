import json
from dataset import *
from model import *
from collections import Mapping
import torch.utils.data
import os
import numpy as np

def dict_update(d, u):
    """ Recursively updates a dictionary with another. Used for parsing settings for training.
    
    Parameters:
    -----------
    d : dict
        The dict to update.
    u : dict
        The dict that contains keys that should be updated in d.
    """
    for key in u:
        if key in d:
            if isinstance(d[key], Mapping):
                dict_update(d[key], u[key])
            else:
                d[key] = u[key]
        else:
            raise RuntimeError(f'Unkown setting {key}')

def dataset_from_config(config):
    """ Creates a dataset from a configuration file. 
    
    Parameters:
    -----------
    config : dict
        The configuration dict.
    
    Returns:
    --------
    train : torch.util.dataset.Dataset
        Training dataset.
    val : torch.util.dataset.Dataset
        Validation dataset.
    test : torch.util.dataset.Dataset
        Testing dataset.
    
    """
    dataset_config = config['dataset']

    dataset = SparseDataset(
        dataset_config.get('directory', os.getenv('ICECUBEGCNDATADIR', '../data')),
        dataset_config.get('file', 'all_energies.hd5'),
        dataset_config['dom_features'],
        num_nearest_neighbours=dataset_config.get('num_nearest_neighbours', 8),
        metric=np.array(dataset_config.get('metric', z_adjusted_metric)),
        idxs=None,
    )

    val_portion = dataset_config.get('validation_portion', 0.1)
    test_portion = dataset_config.get('testing_portion', 0.1)
    assert (test_portion + val_portion) < 1.0
    val_length = int(len(dataset) * val_portion)
    test_length = int(len(dataset) * test_portion)
    train_length = len(dataset) - val_length - test_length

    return torch.utils.data.random_split(dataset, [train_length, val_length, test_length])
    

def model_from_config(config):
    """ Creates a model from a configuration.
    
    Parameters:
    -----------
    config : dict
        The configuration for the model.
    
    Returns:
    --------
    model : torch.nn.model
        A PyTorch model.
    """
    dataset_config = config['dataset']
    model_config = config['model']

    return SparseGCN(
        len(dataset_config['dom_features']),
        model_config['hidden_sizes'],
        model_config['linear_sizes'],
        use_distance_kernel=model_config.get('use_distance_kernel', True),
        use_batchnorm=model_config.get('use_batchnorm', True),
    )

