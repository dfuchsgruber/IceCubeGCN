import torch.nn

def weighted_bce_loss(predictions, targets, weights):
    """ Binary cross entropy loss that applies weights to all instances.
    
    Paramters:
    ----------
    predictions : torch.Tensor, shape [N]
        The logits for each instance.
    targets : torch.Tensor, shape [N]
        The ground truth class labels, each either 0 or 1.
    weigths : torch.Tensor, shape [N]
        The weights for each instance.

    Returns:
    --------
    loss : torch.float
        The binary cross entropy loss over all instances.
    """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    loss *= weights
    return loss.mean()

def weighted_ce_loss(predictions, targets, weights):
    """ Cross entropy loss that applies weights to all instances.
    
    Paramters:
    ----------
    predictions : torch.Tensor, shape [N, num_classes]
        The logits for each sample and class.
    targets : torch.Tensor, shape [N]
        The ground truth class labels, in [0, num_classes[
    weigths : torch.Tensor, shape [N]
        The weights for each instance.

    Returns:
    --------
    loss : torch.float
        The binary cross entropy loss over all instances.
    """
    loss = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
    loss *= weights
    return loss.mean()