import torch
from torch import nn

class AuxiliaryLoss(nn.Module):
    """ Loss that not only considers the classification BCE loss but also a 
    regression loss for the primary particle properties. """

    def __init__(self, lambda_):
        """ Initializes the loss. 
        
        Parameters:
        -----------
        lambda : float
            The weighting of the auxiliary loss term (regression loss).
        """
        self.lambda_ = lambda_
    
    def forward(self, y_pred, y_true, weight=None):
        """ Forward propagation for the auxiliary loss. 
        
        Parameters:
        -----------
        y_pred : tuple
            - class_pred : torch.FloatTensor, shape [batch_size, 1]
                Predicted class probabilities.
            - primary_pred : torch.FloatTensor, shape [batch_size, D]
                Predictions for the properties of the primary praticle.
        y_true : tuple
            - class_true : torch.FloatTensor, shape [batch_size, 1]
                True class labels.
            - primary_true : torch.FloatTensor, shape [batch_size, D]
                True primary particle properties.
        weight : torch.FloatTensor, shape [batch_size, 1] or None
            Optional weights for the samples in the batch.

        Returns:
        --------
        loss : torch.FloatTensor, shape [1]
            The loss for the batch.
        """
        class_pred, primary_pred = y_pred
        class_true, primary_true = y_true
        bce_loss = nn.functional.binary_cross_entropy(class_pred, class_true, weight=weight, reduction='mean')
        if weight is not None:
            mse_loss = torch.mean(weight * nn.functional.mse_loss(primary_pred, primary_true, reduction=None))
        else:
            mse_loss = nn.functional.mse_loss(primary_pred, primary_true, reduction='mean')
        return bce_loss + self.lambda_ * mse_loss
    