import torch
from torch import nn
from .graph import *
from .gcn import GraphConvolutionalNetwork


class GraphConvolutionalNetworkWithGraphFeatures(GraphConvolutionalNetwork):
    """ Subclass of a GCN that also takes graph features and trains a sub-network based on those. """

    def __init__(self, num_input_features_gcn, num_input_features_graph_mlp, units_graph_convolutions = [64, 64, 64, 64, 64], 
        units_fully_connected = [1], units_graph_mlp = [64, 64, 64], dropout_rate=0.5, use_batchnorm=True, use_residual=True, 
        **kwargs):
        """ Creates a GCN model. 

        Parameters:
        -----------
        num_input_features_gcn : int
            Number of features for the input of the GCN part.
        num_input_features_graph_mlp : int
            Number of features for the input of the graph MLP part. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_fully_connected : list or None
            The hidden units for each fully connected layer. If None, the is not built but left for building for sublcasses.
        units_graph_mlp : list
            The hidden units for the layers of the graph feature MLP.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        use_residual : bool
            If the block should implement an additive skip connection. If True, this will only be implemented
            if num_input_features == num_output_features for a graph convolution layer.
        """
        # Embeddings of graph MLP and GCN will be concatenated, thus the input dimension to the Fully Connected output part changes


        super().__init__(num_input_features_gcn, units_graph_convolutions=units_graph_convolutions, 
            units_fully_connected=None, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm, use_residual=use_residual)
        self.layers_graph_mlp = torch.nn.ModuleList()
        for idx, (D, D_prime) in enumerate(zip([num_input_features_graph_mlp] + units_graph_mlp[:-1], units_graph_mlp)):
            is_last_layer = idx == len(units_graph_mlp) - 1
            self.layers_graph_mlp.append(nn.Linear(D, D_prime, bias=True))
            if not is_last_layer and use_batchnorm:
                self.layers_graph_mlp.append(nn.BatchNorm1d(D_prime))
            self.layers_graph_mlp.append(nn.ReLU())
            if not is_last_layer and dropout_rate:
                self.layers_graph_mlp.append(nn.Dropout(dropout_rate))
        
        for idx, (D, D_prime) in enumerate(zip([units_graph_convolutions[-1] + units_graph_mlp[-1]] + units_fully_connected[:-1], units_fully_connected)):
                is_last_layer = idx == len(units_fully_connected) - 1
                self.layers_fully_connected.append(nn.Linear(D, D_prime, bias=True))
                if not is_last_layer:
                    self.layers_fully_connected.append(nn.ReLU())
                else:
                    self.layers_fully_connected.append(nn.Sigmoid())
    

    def forward(self, X, D, masks, F):
        """ Forward pass.
        
        Parameters:
        -----------
        X : torch.FloatTensor, shape [batch_size, N, D]
            The node features.
        D : torch.FloatTensor, shape [batch_size, N, N]
            Pairwise distances for all nodes.
        masks : torch.FloatTensor, shape [batch_size, N, N]
            Masks for the adjacency / distance matrix.
        F : torch.FloatTensor, shape [batch_size, K]
            Graph features that are fed into a separate MLP network.
        """

        # Graph convolutions
        A = self.kernel(D, masks)
        for graph_convolution in self.graph_convolutions:
            X = graph_convolution(X, A, masks)

        # Average pooling
        X = padded_vertex_mean(X, masks, vertex_axis=-2, keepdim=False)

        # Graph feature MLP
        for layer in self.layers_graph_mlp:
            F = layer(F)
        
        # Concatenate graph embeddings with graph feature embeddings
        X = torch.cat([X, F], -1)

        # Fully connecteds, (usually only 1 layer, i.e. logistic regression)
        for layer in self.layers_fully_connected:
            X = layer(X)
        return X
    