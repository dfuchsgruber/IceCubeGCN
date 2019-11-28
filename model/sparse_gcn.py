import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter

class GaussianKernelDistance(torch.nn.Module):
    """ Guassian kernel applied to parwise-distances between connected nodes. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_sigma = torch.nn.Parameter(torch.rand(1) * 0.02 + 0.99)

    def forward(self, edge_idxs, pos):
        """ Forward pass through the gaussian kernel. 
        
        Parameters:
        -----------
        edge_idxs : torch.LongTensor, shape [2, num_edges]
            The edge indices of the graph.
        pos : torch.FloatTensor, shape [num_vertices, num_dimensions]
            The positions of each vertex in a vector space.

        Returns:
        --------
        edge_weights : torch.FloatTensor, shape [num_edges, 1]
            Distances between the nodes after applying a gaussian kernel.
        """
        edge_positions = pos[edge_idxs]
        distances = torch.norm(edge_positions[0] - edge_positions[1], p=2, dim=-1)
        edge_weights = torch.exp(-(self.inverse_sigma * distances**2))
        return edge_weights


class GCNResidualBlock(torch.nn.Module):
    """ Implements a graph convolutional version of the resnet block. """

    def __init__(self, num_input_features, hidden_dim, batchnorm=True):
        """ Initializes the residual gcn block. 
        
        Parameters:
        -----------
        num_input_features : int
            How many features each input vertex has.
        hidden_dim : int
            The output dimension of the convolution.
        batchnorm : bool
            If True, batch normalization is used.
        """
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_input_features, hidden_dim, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.act = torch.nn.ReLU(inplace=True)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        # The residual functionality is only implemented if the input and ouput size are equal
        self.residual = num_input_features == hidden_dim


    def forward(self, x, edge_index, edge_weight=None):
        """ Forward pass.
        
        Parameters:
        -----------
        x : torch.FloatTensor, shape [num_vertices, input_dim]
            The input features for nodes of all graphs in the batch.
        edge_index : torch.LongTensor, shape [2, num_edges]
            The edges of all graphs in the batch.
        edge_weight : torch.LongTensor, shape [num_edges, num_edge_features] or None
            Edge weights for all graphs in the batch.

        Returns:
        --------
        out : torch.FloatTensor, shape [num_vertices, output_dim]
        """
        identity = x
        out = self.conv1(x, edge_index, edge_weight=edge_weight)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out, edge_index, edge_weight=edge_weight)
        out = self.bn2(out)

        if self.residual:
            out += identity
        out = self.act(out)
        return out

class SparseGCN(torch.nn.Module):
    """ Model that operates on sparse graphs. """

    def __init__(self, num_input_features, hidden_dims, linear_dims, use_distance_kernel=True, use_batchnorm=True):
        """ Initializes a GCN model.
        
        Parameters:
        -----------
        num_input_features : int
            The number of features per node.
        hidden_dim : list of int
            A list of hidden representation sizes for each convolution layer.
        linear_dims : list of int
            A list of hidden representation sizes for each linear linear. The last dimension corresponds
            to the number of outputs.
        use_distance_kernel : bool
            If True, the edge weights are given by applying a gaussian kernel to the node distances.
        use_batchnorm : bool
            If True, 1d-batchnormalization is used to normalize mini-batches. 
        """
        super().__init__()
        if use_distance_kernel: 
            self.distance_kernel = GaussianKernelDistance()
        else:
            self.distance_kernel = None
        
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        input_dim = num_input_features
        for hidden_dim in hidden_dims:
            self.convs.append(GCNResidualBlock(input_dim, hidden_dim))
            input_dim = hidden_dim
        for hidden_dim in linear_dims:
            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        
    def forward(self, data):
        """ Forward pass through the network. 
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            The data to pass through the network. All graphs of the batch are aggregated into a single block-wise adjacency.
        
        Returns:
        --------
        scores : torch.FloatTensor, shape [batch_size, num_outputs]
            The scores for each graph.
        """
        x = data.x
        if self.distance_kernel:
            edge_weights = self.distance_kernel(data.edge_index, data.pos)
        else:
            edge_weights = None

        # Graph convolutions
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_weight=edge_weights)

        # Aggregate 
        x = torch_scatter.scatter_mean(x, data.batch, dim=0)

        # Linear
        for idx, linear in enumerate(self.linears):
            x = linear(x)
            if idx != len(self.linears) - 1:
                x = F.relu(x)
        
        return x
