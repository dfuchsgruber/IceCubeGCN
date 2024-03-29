#!/usr/bin/env python3

import os.path
from dataset import *
import util, loss
import numpy as np
import sys
import os.path
import json, pickle
import argparse
from glob import glob
from collections import Mapping, defaultdict
import time

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

import torch
from torch.utils.data import Dataset
import torch_geometric.data
from torch import nn

def log(logfile, string):
    """ Prints a string and puts into the logfile if present. """
    print(string)
    if logfile is not None:
        logfile.write(str(string) + '\n')


def get_metrics(y_true, y_pred):
    """ Calculates all desired metrics for a given set of predictions and ground truths.
    
    Parameters:
    -----------
    y_true : ndarray, shape [N]
        Ground truth class labels.
    y_pred : ndarray, shape [N] or ndarray, shape [N, num_classes]
        Class scores.
        
    Returns:
    --------
    metrics : defaultdict
        A dict containing values for all metrics. 
    """
    metrics = defaultdict(float)
    if len(y_pred.shape) == 1:    
        labels = y_pred >= .5 # Binary classification
        metrics['ppr'] = labels.sum() / y_pred.shape[0]
    else:
        labels = y_pred.argmax(axis=1) # Multiple class classification
    metrics['accuracy'] = accuracy_score(y_true, labels)
    try: 
        metrics['auc'] = roc_auc_score(y_true, labels)
    except Exception as e:
        #print(e) 
        # AUC is not defined, if only one class is present 
        metrics['auc'] = np.nan
    return metrics
    

def evaluate_model(model, data_loader, loss_function, logfile=None):
    """ Evaluates the model performance on a dataset (validation or test).
    
    Parameters:
    -----------
    model : torch.nn.Module
        The classifier to evaluate.
    data_loader : torch.utils.data.DataLoader
        Loader for the dataset to evaluate on.
    loss_function : function
        The loss function that is optimized.
    logfile : file-like or None
        The file to put logs into.

    Returns:
    --------
    metrics : defaultdict(float)
        The statistics (metrics) for the model on the given dataset.
    """
    model.eval()
    metrics = defaultdict(float)
    number_classes = data_loader.dataset.get_number_classes()
    if number_classes == 2:
        y_pred = np.zeros(len(data_loader.dataset))
    else:
        y_pred = np.zeros((len(data_loader.dataset), number_classes))
    y_true = np.zeros(len(data_loader.dataset))
    total_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = data.to('cuda')
            print(f'\rEvaluating {batch_idx + 1} / {len(data_loader)}', end='\r')
            y_pred_i = model(data)
            loss = loss_function(y_pred_i, data.y, data.weight)
            y_pred[batch_idx * data_loader.batch_size : (batch_idx + 1) * data_loader.batch_size] = y_pred_i.data.cpu().numpy().squeeze()
            y_true[batch_idx * data_loader.batch_size : (batch_idx + 1) * data_loader.batch_size] = data.y.data.cpu().numpy().squeeze()
            total_loss += loss.item()

    metrics = get_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / len(data_loader)
    
    print(metrics)
    values = ' -- '.join(map(lambda metric: f'{metric} : {(metrics[metric]):.4f}', metrics))
    log(logfile, f'\nMetrics: {values}')
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Configuration for the training. See default_settings.json for the default settings. Values are updated with the settings passed here.')
    parser.add_argument('--array', help='If set, the "config" parameter refers to a regex. Needs the file index parameter.', action='store_true')
    parser.add_argument('-i', type=int, help='Index of the file in the directory to use as configuration file. Only considered if "--array" is set.')
    args = parser.parse_args()


    if args.array:
        config_path = glob(args.config)[args.i]
    else:
        config_path = args.config
    
    with open(config_path) as f:
        settings = json.load(f)

    # Create a logfile
    if settings['training']['logfile']:
        logfile = open(settings['training']['logfile'], 'w+')
    else:
        logfile = None

    log(logfile, f'### Training according to configuration {config_path}')

    # Set up the directory for training and saving the model
    model_idx = np.random.randint(10000000000)
    log(logfile, f'### Generating a model id: {model_idx}')
    training_dir = settings['training']['directory'].format(model_idx)
    log(logfile, f'### Saving to {training_dir}')
    os.makedirs(training_dir, exist_ok=True)

    # Create a seed if non given
    if settings['seed'] is None:
        settings['seed'] = model_idx
        print(f'Seeded with the model id ({model_idx})')

    np.random.seed(settings['seed'] & 0xFFFFFFFF)
    torch.manual_seed(settings['seed'] & 0xFFFFFFFF)

    # Save a copy of the settings
    with open(os.path.join(training_dir, 'config.json'), 'w+') as f:
        json.dump(settings, f)
    
    # Load data
    batch_size = settings['training']['batch_size']

    data_train, data_val, data_test = util.dataset_from_config(settings)
    train_loader = torch_geometric.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = torch_geometric.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = torch_geometric.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, drop_last=False)

    model = util.model_from_config(settings)
    if torch.cuda.is_available():
        model = model.cuda()
        log(logfile, "Training on GPU")
        log(logfile, "GPU type:\n{}".format(torch.cuda.get_device_name(0)))
    if settings['training']['loss'].lower() == 'binary_crossentropy':
        loss_function = loss.weighted_bce_loss
    elif settings['training']['loss'].lower() == 'categorical_cross_entropy':
        loss_function = loss.weighted_ce_loss
    else:
        raise RuntimeError(f'Unkown loss {settings["training"]["loss"]}')

    optimizer = torch.optim.Adamax(model.parameters(), lr=settings['training']['learning_rate'])
    lr_scheduler_type = settings['training']['learning_rate_scheduler']
    if lr_scheduler_type.lower() == 'reduce_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
        patience=settings['training']['learning_rate_scheduler_patience'], min_lr=settings['training']['min_learning_rate'])
    elif lr_scheduler_type.lower() == 'exponential_decay':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, settings['training']['learning_rate_decay'])
    elif lr_scheduler_type.lower() == 'constant':
        lr_scheduler = None
    else:
        raise RuntimeError(f'Unkown learning rate scheduler strategy {lr_scheduler_type}')

    validation_metrics = defaultdict(list)
    training_metrics = defaultdict(list)

    log(logfile, f'Training on {len(data_train)} samples.')

    epochs = settings['training']['epochs']
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}, learning rate: {optimizer.param_groups[0]["lr"]}')
        running_loss = 0
        running_accuracy = 0
        model.train()
        t0 = time.time()
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to('cuda')
            y_pred = model(data)
            loss = loss_function(y_pred, data.y, data.weight)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            targets = data.y.cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

            # Calculate the variance accross predictions to ensure that the model does not overfit a single-class output
            if len(y_pred.shape) == 1:    
                y_pred_labels = y_pred >= .5 # Binary classification
                metrics['ppr'] = labels.sum() / y_pred.shape[0]
            else:
                y_pred_labels = y_pred.argmax(axis=1) # Multiple class classification

            batch_metrics = get_metrics(targets, y_pred)
            for metric, value in batch_metrics.items():
                training_metrics[metric].append(value)
            running_accuracy += batch_metrics['accuracy']
            # Estimate ETA
            dt = time.time() - t0
            eta = dt * (len(train_loader) / (batch_idx + 1) - 1)

            print(f'\r{batch_idx + 1} / {len(train_loader)}: batch_loss {loss.item():.4f} -- epoch_loss {running_loss / (batch_idx + 1):.4f} -- epoch acc {running_accuracy / (batch_idx + 1):.4f} -- std of predicted labels {y_pred_labels.std():.4f} -- # ETA: {int(eta):6}s      ', end='\r')

        # Validation
        log(logfile, '\n### Validation:')    
        for metric, value in evaluate_model(model, val_loader, loss_function, logfile=logfile).items():
            validation_metrics[metric].append(value)
        # Update learning rate, scheduler uses last accuracy as cirterion
        if lr_scheduler:
            lr_scheduler.step(validation_metrics['accuracy'][-1])

        # Save model parameters
        checkpoint_path = os.path.join(training_dir, f'model_{epoch + 1}')
        torch.save(model.state_dict(), checkpoint_path)
        log(logfile, f'Saved model to {checkpoint_path}')
    
    log(logfile, '\n### Testing:')
    testing_metrics = evaluate_model(model, test_loader, loss_function, logfile=logfile)

    with open(os.path.join(training_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump({'train': training_metrics, 'val' : validation_metrics, 'test' : testing_metrics}, f)

    if logfile is not None:
        logfile.close()





