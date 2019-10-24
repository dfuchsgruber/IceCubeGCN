import torch
from torch import nn
from .gcn_with_graph_features import GraphConvolutionalNetworkWithGraphFeatures

class AuxiliaryGraphConvolutionalNetworkWithGraphFeatures(GraphConvolutionalNetworkWithGraphFeatures):
    """ Model that uses graph features as well as auxiliary learning to predict tracks. """

    def __init__(self, num_input_features_gcn, num_input_features_graph_mlp, units_graph_convolutions = [64, 64, 64, 64, 64], 
        units_fully_connected_classification = [1], units_fully_connected_regression = [5], units_graph_mlp = [64, 64, 64], 
        dropout_rate=0.5, use_batchnorm=True, 
        use_residual=True, **kwargs):
        """ Creates a GCN model. 

        Parameters:
        -----------
        num_input_features_gcn : int
            Number of features for the input of the GCN part.
        num_input_features_graph_mlp : int
            Number of features for the input of the graph MLP part. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_fully_connected_classification : list or None
            The hidden units for each fully connected layer of the classification task. 
            If None, the layers are not built, i.e. a subclass has to build them.
        units_fully_connected_regression : list or None
            The hidden units for each fully connected layer of the regression task.
            If None, the layers are not built, i.e. a subclass has to build them.
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
        super().__init__(num_input_features_gcn, num_input_features_graph_mlp, 
            units_graph_convolutions=units_graph_convolutions, units_fully_connected=units_fully_connected_classification, 
            units_graph_mlp=units_graph_mlp, use_batchnorm=use_batchnorm, use_residual=use_residual, **kwargs)
        # Build the regression network
        if units_fully_connected_regression is not None:
            self.layers_regression = torch.nn.ModuleList()
            for idx, (D, D_prime) in enumerate(zip(
                [units_graph_convolutions[-1] + units_graph_mlp[-1]] + units_fully_connected_regression[:-1], units_fully_connected_regression)):
                is_last_layer = idx == len(units_fully_connected) - 1
                self.layers_regression.append(nn.Linear(D, D_prime, bias=True))
                if not is_last_layer:
                    self.layers_regression.append(nn.ReLU())

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
        
        Returns:
        --------
        y_classification : torch.FloatTensor, shape [batch_size, units_fully_connected_classifcation[-1]]
            Classification outputs.
        y_regression : torch.FloatTensor, shape [batch_size, units_fully_connected_regression[-1]]
            Regression outputs.
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
        y_classification = X
        for layer in self.layers_fully_connected:
            y_classification = layer(y_classification)

        # Fully connecteds for regression task
        y_regression = X
        for layer in self.layers_regression:
            y_regression = layer(y_regression)
        
        return y_classification, y_regression