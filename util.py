import json
from dataset import *
from model import *
from collections import Mapping



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

def dataset_from_config(config, filter_non_train=False, close_file=True):
    """ Creates a dataset from a configuration file. 
    
    Parameters:
    -----------
    config : dict
        The configuration dict.
    filter_non_train : bool
        If true, the validation and testing data set are filtered.
    close_file : bool
        If True, hd5 files will be closed after readout.
    
    Returns:
    --------
    train : dataset.ShuffledTorchHD5Dataset
        Training dataset.
    val : dataset.ShuffledTorchHD5Dataset
        Validation dataset.
    test : dataset.ShuffledTorchHD5Dataset
        Testing dataset.
    
    """
    dataset_config = config['dataset']
    dataset_type = dataset_config['type'].lower()
    if dataset_type in ('hdf5', 'hd5'):
        train = ShuffledGraphTorchHD5Dataset(
            dataset_config['paths']['train'],
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            class_weights = dataset_config['class_weights'],
            close_file = close_file,
            )
        val = ShuffledGraphTorchHD5Dataset(
            dataset_config['paths']['validation'],
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            class_weights = dataset_config['class_weights'],
            close_file = close_file,
            )
        test = ShuffledGraphTorchHD5Dataset(
            dataset_config['paths']['test'],
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            class_weights = dataset_config['class_weights'],
            close_file = close_file,
            )
        return train, val, test
    elif dataset_type in ('hdf5_graph_features', 'hd5_graph_features'):
        train = ShuffledGraphTorchHD5DatasetWithGraphFeatures(
            dataset_config['paths']['train'],
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            graph_features = dataset_config['graph_features'],
            class_weights = dataset_config['class_weights'],
            close_file = close_file,
            )
        val = ShuffledGraphTorchHD5DatasetWithGraphFeatures(
            dataset_config['paths']['validation'],
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            graph_features = dataset_config['graph_features'],
            class_weights = dataset_config['class_weights'],
            close_file = close_file,
            )
        test = ShuffledGraphTorchHD5DatasetWithGraphFeatures(
            dataset_config['paths']['test'],
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            graph_features = dataset_config['graph_features'],
            class_weights = dataset_config['class_weights'],
            close_file = close_file,
            )
        return train, val, test
    else:
        raise RuntimeError(f'Unknown dataset type {dataset_type}')

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
    number_input_features = len(config['dataset']['features'])
    model_config = config['model']
    model_type = model_config['type'].lower()
    if model_type in ('gcn', 'gcnn'):
        model = GraphConvolutionalNetwork(
            number_input_features,
            units_graph_convolutions = model_config['hidden_units_graph_convolutions'],
            units_fully_connected = model_config['hidden_units_fully_connected'],
            use_batchnorm = model_config['use_batchnorm'],
            dropout_rate = model_config['dropout_rate'],
            use_residual = model_config['use_residual'],
        )
        num_classes = (model_config['hidden_units_graph_convolutions'] + model_config['hidden_units_fully_connected'])[-1]
    elif model_type in ('gcn_graph_features', 'gcnn_graph_features'):
        model = GraphConvolutionalNetworkWithGraphFeatures(
            number_input_features,
            len(config['dataset']['graph_features']),
            units_graph_convolutions = model_config['hidden_units_graph_convolutions'],
            units_fully_connected = model_config['hidden_units_fully_connected'],
            units_mlp = model_config['hidden_units_graph_mlp'],
            use_batchnorm = model_config['use_batchnorm'],
            dropout_rate = model_config['dropout_rate'],
            use_residual = model_config['use_residual'],
        )
    elif model_type in ('gcn_graph_features_regression', 'gcnn_graph_features_regression'):
        model = AuxiliaryGraphConvolutionalNetworkWithGraphFeatures(
            number_input_features,
            len(config['dataset']['graph_features']),
            units_graph_convolutions = model_config['hidden_units_graph_convolutions'],
            units_fully_connected_classification = model_config['hidden_units_fully_connected'],
            units_fully_connected_regression = model_config['hidden_units_regression'],
            units_mlp = model_config['hidden_units_graph_mlp'],
            use_batchnorm = model_config['use_batchnorm'],
            dropout_rate = model_config['dropout_rate'],
            use_residual = model_config['use_residual'],
        )
    else:
        raise RuntimeError(f'Unkown model type {model_type}')
    return model