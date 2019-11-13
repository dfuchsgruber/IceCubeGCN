# Sparse dataset class for IceCube graphs
# Each DOM is only connected to its k nearest neighbours

import h5py
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
from sklearn.neighbors import kneighbors_graph
import os.path

horizontal_dom_spacing = 125
vertical_dom_spacing = 17

class SparseDataset(Dataset):

    def __init__(self, root, path, dom_features, *args, num_nearest_neighbours=8, vertical_scaling=horizontal_dom_spacing / vertical_dom_spacing, **kwargs):
        """ Initializes the sparse dataset. 
        
        Parameters:
        -----------
        path : str
            A path to a hd5 file, containing all events that are to be considered.
        dom_features : dict
            A dict mapping per-dom feature columns of the hd5 data to scaling factors.
        num_nearest_neighbours : int
            How many DOMs any DOM is connected to.
        vertical_scaling : float
            All distances in Z-direction are scaled using this factor, to ensure that DOMs are not only
            connected along the vertical axis.
        """
        super().__init__(root, *args, **kwargs)
        self.dom_features = dom_features
        self.num_nearest_neighbours = num_nearest_neighbours

        self.file = h5py.File(os.path.join(root, path), 'r')
        self.number_vertices = np.array(self.file['NumberVertices'])
        self.offsets = self.number_vertices.cumsum() - self.number_vertices
        self.targets = self._create_targets()

        self.distance_metric = np.diag([1.0, 1.0, vertical_scaling]) # Distance metric for the doms

    def _create_targets(self):
        """ Creates targets for classification based on some filtered indices. 
        
        Returns:
        --------
        targets : ndarray, shape [N]
            Class labels.
        """
        targets = np.array(self.file['classification'], dtype=np.int)
        targets[targets == 22] = 2
        targets[targets == 23] = 4
        targets[targets == 11] = 0
        assert len(np.unique(targets)) == 5
        return targets
    
    def __len__(self):
        return self.number_vertices.shape[0]

    def get(self, idx):
        """ Gets a graph instance from the dataset. """
        offset = self.offsets[idx]
        number_vertices = self.number_vertices[idx]
        node_features = np.array([
            self.file[feature][offset : offset + number_vertices] * scaling for (feature, scaling) in self.dom_features.items()
        ]).T
        node_coordinates = np.array([
            self.file[coordinate][offset : offset + number_vertices] for coordinate in ('VertexX', 'VertexY', 'VertexZ')
        ]).T
        edges = kneighbors_graph(
            node_coordinates, min(number_vertices - 1, self.num_nearest_neighbours), metric='mahalanobis', metric_params={'V' : self.distance_metric}
            ).nonzero()
        return Data(x=node_features, edge_index=np.array(edges), pos=node_coordinates, y=self.targets[idx])
        

    def _download(self): 
        pass

    def _process(self):
        pass
