
import torch
import torch.nn as nn
from .graph import *

class GraphConvolutionalNetwork(nn.Module):
    """ Module that first performs graph convolutions, use an average pooling on the node embeddings and finally a MLP. """
    def __init__(self, num_input_features, units_graph_convolutions = [64, 64, 64, 64, 64], units_fully_connected = [1], 
        dropout_rate=0.5, use_batchnorm=True, use_residual=True, **kwargs):
        """ Creates a GCN model. 

        Parameters:
        -----------
        num_input_features : int
            Number of features for the input. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_fully_connected : list or None
            The hidden units for each fully connected layer. Pass '1' for logistic regression of pooled node features.
            If None, a subclass must build the fully connected layers.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        use_residual : bool
            If the block should implement an additive skip connection. If True, this will only be implemented
            if num_input_features == num_output_features for a graph convolution layer.
        """
        super().__init__(**kwargs)
        self.kernel = GaussianKernel()
        self.graph_convolutions = torch.nn.ModuleList()
        self.layers_fully_connected = torch.nn.ModuleList()
        for idx, (D, D_prime) in enumerate(zip([num_input_features] + units_graph_convolutions[:-1], units_graph_convolutions)):
            is_last_layer = idx == len(units_graph_convolutions) - 1
            self.graph_convolutions.append(
                GraphConvolution(
                    D, D_prime, use_bias=True, 
                    activation=not is_last_layer, 
                    dropout_rate=None if is_last_layer else dropout_rate,
                    use_batchnorm=use_batchnorm and not is_last_layer
                    ))

        if units_fully_connected is not None:
            for idx, (D, D_prime) in enumerate(zip([units_graph_convolutions[-1]] + units_fully_connected[:-1], units_fully_connected)):
                is_last_layer = idx == len(units_fully_connected) - 1
                self.layers_fully_connected.append(nn.Linear(D, D_prime, bias=True))
                if not is_last_layer:
                    self.layers_fully_connected.append(nn.ReLU())
                else:
                    self.layers_fully_connected.append(nn.Sigmoid())
        
    def forward(self, X, D, masks):
        """ Forward pass.
        
        Parameters:
        -----------
        X : torch.FloatTensor, shape [batch_size, N, D]
            The node features.
        D : torch.FloatTensor, shape [batch_size, N, N]
            Pairwise distances for all nodes.
        masks : torch.FloatTensor, shape [batch_size, N, N]
            Masks for the adjacency / distance matrix.
        """

        # Graph convolutions
        A = self.kernel(D, masks)
        for graph_convolution in self.graph_convolutions:
            X = graph_convolution(X, A, masks)

        # Average pooling
        X = padded_vertex_mean(X, masks, vertex_axis=-2, keepdim=False)

        # Fully connecteds, (usually only 1 layer, i.e. logistic regression)
        for layer in self.layers_fully_connected:
            X = layer(X)
        return X
    
            