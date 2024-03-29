# Sparse dataset class for IceCube graphs
# Each DOM is only connected to its k nearest neighbours

import h5py
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.class_weight import compute_sample_weight
import os.path
from time import time

horizontal_dom_spacing = 125
vertical_dom_spacing = 17

# This metric favours connections in the x-y plane, since the DOMs are populated more densly on the z axis
z_adjusted_metric = np.diag([1.0, 1.0, horizontal_dom_spacing / vertical_dom_spacing])

class SparseDataset(Dataset):

    def __init__(self, root, path, dom_features, *args, num_nearest_neighbours=8, idxs=None, **kwargs):
        """ Initializes the sparse dataset. 
        
        Parameters:
        -----------
        root : str
            The directory where data is located at.
        path : str
            A path to a hd5 file, relative to the data directory of the `root` parameter,
            containing all events that are to be considered.
        dom_features : OrderedDict
            A dict mapping per-dom feature columns of the hd5 data to scaling factors.
        num_nearest_neighbours : int
            How many DOMs any DOM is connected to.
        idxs : ndarray, shape [N] or str or None
            A subset of indices that are considered. This is useful if the data is preprocessed.
        """
        super().__init__(root, *args, **kwargs)
        self.dom_features = dom_features
        self.num_nearest_neighbours = num_nearest_neighbours

        self.file = h5py.File(os.path.join(root, path), 'r')
        self.number_vertices = np.array(self.file['NumberVertices'])
        self.offsets = self.number_vertices.cumsum() - self.number_vertices
        self.targets = self._create_targets()

        if idxs is None:
            self.idxs = np.arange(self.targets.shape[0])
        elif isinstance(idxs, str):
            self.idxs = np.load(idxs)
        elif isinstance(idxs, np.ndarray):
            self.idxs = idxs
        else:
            raise RuntimeError(f'Indices of the data must either be of type `str` or `ndarray` or `None`, not {type(idxs)}')

        assert self.idxs.shape[0] == self.targets.shape[0]

        # Estimate sample weights, by only considering instances which are part of self.idxs
        self.weights = np.zeros(self.targets.shape[0])
        self.weights[self.idxs] = compute_sample_weight('balanced', self.targets[self.idxs])

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
        return self.idxs.shape[0]


    def get(self, dataset_idx):
        """ Gets a graph instance from the dataset. """
        idx = self.idxs[dataset_idx]
        offset = self.offsets[idx]
        number_vertices = self.number_vertices[idx]

        node_features = torch.tensor([
            self.file[feature][offset : offset + number_vertices] * scaling for (feature, scaling) in self.dom_features.items()
        ]).T
        node_coordinates = torch.tensor([
            self.file[coordinate][offset : offset + number_vertices] for coordinate in ('VertexX', 'VertexY', 'VertexZ')
        ]).T
        adjacency_list = np.array(self.file['AdjacencyList'][offset : offset + number_vertices])[:, : self.num_nearest_neighbours]
        edges = np.array([adjacency_list.flatten(), np.arange(adjacency_list.shape[0]).repeat(adjacency_list.shape[1])], dtype=np.int16)
        return Data(x=node_features, edge_index=torch.tensor(edges, dtype=torch.long), pos=node_coordinates, 
                    y=torch.tensor([self.targets[idx]], dtype=torch.long), weight=torch.tensor([self.weights[idx]]))
        

    def _download(self): 
        pass

    def _process(self):
        pass


def create_memmap_1d(memmap_file, hd5_file, features, dtype=np.float64):
    """ Creates a memmap of 1d features or loads the memmap if it already exists.
    
    Parameters:
    -----------
    memmap_file : str
        Path to the memmap file to create.
    hd5_file : h5py.File
        The hd5f file that contains data.
    features : iterable
        An iterable of feature columns to copy.
    dtype : np.dtype
        The numpy datatype.

    Returns:
    --------
    memmap : np.memmap, shape [N, len(features)]
        Feature matrix.
    """
    # Check that sizes of all feature columns match
    feature_sizes = np.array([hd5_file[key].shape for key in features])
    assert (feature_sizes[0] == feature_sizes).all()

    if os.path.exists(memmap_file):
        memmap = np.memmap(memmap_file, 'r', shape=(feature_sizes[0], len(features)), dtype=dtype)
    else:
        # Create a new memmap
        print(f'Creating memmap {memmap_file}...')
        memmap = np.memmap(memmap_file, 'w+', shape=(feature_sizes[0], len(features)), dtype=dtype)
        for idx, key in enumerate(features):
            print(f'\rCopying feature {feature}...', end='\r')
            memmap[:, idx] = hd5_file.get(feature)
    return memmap

def create_memmap_2d(memmap_file, hd5_file, feature, dtype=np.int16):
    """ Creates a memmap from a 2d feature matrix in a hd5 file.
    
    Parameters:
    -----------
    memmap_file : str
        Path to the memmap file to create.
    hd5_file : h5py.File
        The hd5f file that contains data.
    feature : str
        The feature that contains a 2d data matrix.
    dtype : np.dtype
        The numpy datatype.

    Returns:
    --------
    memmap : np.memmap, shape [N, D]
        The 2d feature matrix as numpy memmap.
    """
    assert len(hd5_file[feature].shape) == 2

    if os.path.exists(memmap_file):
        memmap = np.memmap(memmap_file, 'r', shape=hd5_file[feature].shape, dtype=dtype)
    else:
        # Create a new memmap
        print(f'Creating memmap {memmap_file}...')
        memmap = np.memmap(memmap_file, 'w+', shape=hd5_file[feature].shape, dtype=dtype)
        for idx in range(hd5_file[feature].shape[1]):
            print(f'\rCopying column {idx}...', end='\r')
            memmap[:, idx] = hd5_file[feature][:, idx]
    return memmap