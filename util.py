import json
from dataset import *
from model import *
from collections import Mapping, OrderedDict
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
    data_train, data_val, data_test = [
        SparseDataset(
            dataset_config.get('directory', os.getenv('ICECUBEGCNDATADIR', '../data')),
            dataset_config[key],
            OrderedDict(dataset_config['dom_features']),
            num_nearest_neighbours=dataset_config.get('num_nearest_neighbours', 8),
            idxs=None,
        ) for key in ('train', 'val', 'test') 
    ]
    return data_train, data_val, data_test
    

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

