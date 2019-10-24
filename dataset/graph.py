from .hd5 import *


class ShuffledGraphTorchHD5Dataset(ShuffledTorchHD5Dataset):
    """ Class to represent a pre-shuffled PyTorch dataset originating from an HD5File. """

    def __init__(self, filepath, features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'], 
        memmap_directory='./memmaps', close_file=True, 
        class_weights=None):
        """ Initializes the Dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        features : list
            A list of dataset columns in the HD5 File that represent the vertex features.
        coordinates : list
            A list of coordinate columns in the HD5 File that represent the coordinates of each vertex.
        memmap_directory : str
            Directory for memmaps.
        close_file : bool
            If True, the hd5 file will be closed afterwards.
        class_weights : dict or None
            Weights for each class.
        """
        super().__init__(filepath)
        self.feature_names = features
        self.coordinate_names = coordinates

        self.targets = self._create_targets()
        self.number_vertices = np.array(self.file['NumberVertices'])
        self.event_offsets = self.number_vertices.cumsum() - self.number_vertices

        # Create memmaps for features and coordinates for faster access during training
        os.makedirs(os.path.dirname(memmap_directory), exist_ok=True)

        # Load precomputed memmaps based on the hash of the columns, filename and index set
        features_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + features
        ]).encode()).hexdigest()
        coordinates_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + coordinates
        ]).encode()).hexdigest()
        print(f'Feature hash {features_hash}')
        print(f'Coordinate hash {coordinates_hash}')

        feature_memmap_path = os.path.join(memmap_directory, f'hd5_features_{features_hash}')
        coordinate_memmap_path = os.path.join(memmap_directory, f'hd5_coordinates_{coordinates_hash}')

        self.features = self._create_feature_memmap(feature_memmap_path, self.feature_names)
        self.coordinates = self._create_feature_memmap(coordinate_memmap_path, self.coordinate_names)
        self.weights = self._compute_weights(class_weights, self.targets)

        # Sanity checks
        endpoints = self.number_vertices + self.event_offsets
        assert(np.max(endpoints)) <= self.features.shape[0]
        # print(np.unique(np.vstack((np.abs(pdg_encoding[self._idxs]), interaction_type[self._idxs])), return_counts=True, axis=1))

        if close_file:
            self.file.close()
            self.file = filepath

    def __len__(self):
        return self.number_vertices.shape[0]
    
    def __getitem__(self, idx):
        N = self.number_vertices[idx]
        offset = self.event_offsets[idx]
        X = self.features[offset : offset + N]
        C = self.coordinates[offset : offset + N]
        return X, C, self.targets[idx], self.weights[idx]

    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        inputs : tuple
            - X : torch.FloatTensor, shape [batch_size, N, D]
                The vertex features.
            - C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
                Pariwise distances.
            - masks : torch.FloatTensor, shape [batch_size, N, N]
                Adjacency matrix masks.
        outputs : tuple
            - torch.FloatTensor, shape [batch_size, N, 1]
                Class labels.
        weights : torch.FloatTensor, shape [batch_size] or None
            Weights for each sample.
        """
        X, C, y, w = zip(*samples)

        # Pad the batch
        batch_size = len(X)
        max_number_vertices = max(map(lambda features: features.shape[0], X))
        features = np.zeros((batch_size, max_number_vertices, X[0].shape[1]))
        coordinates = np.zeros((batch_size, max_number_vertices, C[0].shape[1]))
        masks = np.zeros((batch_size, max_number_vertices, max_number_vertices))
        for idx, (X_i, C_i) in enumerate(zip(X, C)):
            features[idx, : X_i.shape[0]] = X_i
            coordinates[idx, : C_i.shape[0], : C_i.shape[1]] = C_i
            masks[idx, : X_i.shape[0], : X_i.shape[0]] = 1
        
        # Make torch tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        features = torch.FloatTensor(features).to(device)
        coordinates = torch.FloatTensor(coordinates).to(device)
        masks = torch.FloatTensor(masks).to(device)
        targets = torch.FloatTensor(y).to(device).unsqueeze(1)
        weights = torch.FloatTensor(w).to(device).unsqueeze(1)
        return (features, coordinates, masks), targets, weights


class ShuffledGraphTorchHD5DatasetWithGraphFeatures(ShuffledGraphTorchHD5Dataset):
    """ Pre-shuffled PyTorch dataset that also outputs graph features. """
    def __init__(self, filepath, features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'],
        graph_features = ['RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith'], 
        memmap_directory='./memmaps', close_file=True, class_weights=None):
        """ Initializes the Dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        features : list
            A list of dataset columns in the HD5 File that represent the vertex features.
        coordinates : list
            A list of coordinate columns in the HD5 File that represent the coordinates of each vertex.
        graph_features : list
            A list of graph feature columns in the HD5 File that represent event features.
        balance_dataset : bool
            If the dataset should be balanced such that each class contains the same number of samples.
        min_track_length : float or None
            Minimal track length all track-events must have.
        max_cascade_energy : float or None
            The maximal cascade energy all track events are allowed to have.
        flavors : list or None
            Only certain neutrino flavor events will be considered if given.
        currents : list or None
            Only certain current events will be considered if given. 
        memmap_directory : str
            Directory for memmaps.
        close_file : bool
            If True, the hd5 file will be closed after readout.
        class_weights : dict or None
            Weights for each class.
        """
        super().__init__(filepath, features=features, coordinates=coordinates,
            memmap_directory=memmap_directory, 
            close_file=False, class_weights=class_weights)
        self.graph_feature_names = graph_features

        graph_features_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + graph_features
        ]).encode()).hexdigest()

        graph_features_memmap_path = os.path.join(memmap_directory, f'hd5_graph_features_{graph_features_hash}_{self._idxs_hash}')
        self.graph_features = self._create_feature_memmap(graph_features_memmap_path, self._idxs.tolist(), self.graph_feature_names, number_samples=self._idxs.shape[0])

        if close_file:
            self.file.close()
            self.file = filepath

    def __getitem__(self, idx):
        X, C, y, w = super().__getitem__(idx)
        return X, C, self.graph_features[idx], y, w

    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        inputs : tuple
            - X : torch.FloatTensor, shape [batch_size, N, D]
                The vertex features.
            - C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
                Pariwise distances.
            - masks : torch.FloatTensor, shape [batch_size, N, N]
                Adjacency matrix masks.
            - F : torch.FloatTensor, shape [batch_size, N]
                Graph features.
        outputs : tuple
            - targets : torch.FloatTensor, shape [batch_size, N, 1]
                Class labels.
        weights : torch.FloatTensor, shape [batch_size] or None
            Sample weights
        """
        X, C, F, y, w = zip(*samples)

        # Pad the batch
        batch_size = len(X)
        max_number_vertices = max(map(lambda features: features.shape[0], X))
        features = np.zeros((batch_size, max_number_vertices, X[0].shape[1]))
        coordinates = np.zeros((batch_size, max_number_vertices, C[0].shape[1]))
        masks = np.zeros((batch_size, max_number_vertices, max_number_vertices))
        graph_features = np.array(F)
        for idx, (X_i, C_i) in enumerate(zip(X, C)):
            features[idx, : X_i.shape[0]] = X_i
            coordinates[idx, : C_i.shape[0], : C_i.shape[1]] = C_i
            masks[idx, : X_i.shape[0], : X_i.shape[0]] = 1
        
        # Make torch tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        features = torch.FloatTensor(features).to(device)
        coordinates = torch.FloatTensor(coordinates).to(device)
        masks = torch.FloatTensor(masks).to(device)
        targets = torch.FloatTensor(y).to(device).unsqueeze(1)
        graph_features = torch.FloatTensor(graph_features).to(device)
        weights = torch.FloatTensor(w).to(device).unsqueeze(1)
        return (features, coordinates, masks, graph_features), targets, weights

class ShuffledGraphTorchHD5DatasetWithGraphFeaturesAndAuxiliaryTargets(ShuffledGraphTorchHD5DatasetWithGraphFeatures):
    """ Pre-shuffled PyTorch dataset that will yield graph features as well as regression targets for an auxilliary task. """
    def __init__(self, filepath, features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'],
        graph_features = ['RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith'], auxiliary_targets=[],
        balance_dataset=False, min_track_length=None, max_cascade_energy=None, flavors=None, currents=None,
        memmap_directory='./memmaps', close_file=True, class_weights=None):
        """ Initializes the Dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        features : list
            A list of dataset columns in the HD5 File that represent the vertex features.
        coordinates : list
            A list of coordinate columns in the HD5 File that represent the coordinates of each vertex.
        graph_features : list
            A list of graph feature columns in the HD5 File that represent event features.
        auxiliary_targets : list
            A list of graph feature columns that are used as targets for the auxiliary regression task.
        balance_dataset : bool
            If the dataset should be balanced such that each class contains the same number of samples.
        min_track_length : float or None
            Minimal track length all track-events must have.
        max_cascade_energy : float or None
            The maximal cascade energy all track events are allowed to have.
        flavors : list or None
            Only certain neutrino flavor events will be considered if given.
        currents : list or None
            Only certain current events will be considered if given. 
        memmap_directory : str
            Directory for memmaps.
        close_file : bool
            If True, the hd5 will be closed after readout.
        class_weights : dict or None
            Weights for each class.
        """
        super().__init__(filepath, features=features, coordinates=coordinates, balance_dataset=balance_dataset, 
            min_track_length=min_track_length, max_cascade_energy=max_cascade_energy, flavors=flavors, currents=currents,
            memmap_directory=memmap_directory, 
            graph_features=graph_features, close_file=False, class_weights=class_weights)

        self.auxiliary_target_names = auxiliary_targets

        auxiliary_targets_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + auxiliary_targets
        ]).encode()).hexdigest()

        auxiliary_targets_memmap_path = os.path.join(memmap_directory, f'hd5_auxiliary_targets_{auxiliary_targets_hash}_{self._idxs_hash}')
        self.auxiliary_targets = self._create_feature_memmap(auxiliary_targets_memmap_path, self._idxs, self.auxiliary_target_names, self._idxs.shape[0])
        
        if close_file:
            self.file.close()
            self.file = filepath

    def __getitem__(self, idx):
        X, C, F, y, w = super().__getitem__(idx)
        return X, C, F, y, self.auxiliary_targets[idx], w

    
    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        inputs : tuple
            - X : torch.FloatTensor, shape [batch_size, N, D]
                The vertex features.
            - C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
                Pariwise distances.
            - masks : torch.FloatTensor, shape [batch_size, N, N]
                Adjacency matrix masks.
            - F : torch.FloatTensor, shape [batch_size, N]
                Graph features.
        targets : tuple
            - targets : tuple
                - y : torch.FloatTensor, shape [batch_size, 1]
                    Class labels.
                - r : torch.FloatTensor, shape [batch_size, K]
                    Regression targets.
            - w : torch.FloatTensor, shape [batch_size]
                Sample weights.
        """
        X, C, F, y, r, w  = zip(*samples)

        # Pad the batch
        batch_size = len(X)
        max_number_vertices = max(map(lambda features: features.shape[0], X))
        features = np.zeros((batch_size, max_number_vertices, X[0].shape[1]))
        coordinates = np.zeros((batch_size, max_number_vertices, C[0].shape[1]))
        masks = np.zeros((batch_size, max_number_vertices, max_number_vertices))
        graph_features = np.array(F)
        for idx, (X_i, C_i) in enumerate(zip(X, C)):
            features[idx, : X_i.shape[0]] = X_i
            coordinates[idx, : C_i.shape[0], : C_i.shape[1]] = C_i
            masks[idx, : X_i.shape[0], : X_i.shape[0]] = 1
        
        # Make torch tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        features = torch.FloatTensor(features).to(device)
        coordinates = torch.FloatTensor(coordinates).to(device)
        masks = torch.FloatTensor(masks).to(device)
        targets = torch.FloatTensor(y).to(device).unsqueeze(1)
        regression_targets = torch.FloatTensor(r).to(device)
        graph_features = torch.FloatTensor(graph_features).to(device)
        weights = torch.FloatTensor(w).to(device).unsqueeze(1)
        return (features, coordinates, masks, graph_features), (targets, regression_targets), weights