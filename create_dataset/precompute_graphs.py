import h5py
import numpy as np
from sklearn.neighbors import kneighbors_graph
import sys
from tqdm import tqdm

horizontal_dom_spacing = 125
vertical_dom_spacing = 17

# This metric favours connections in the x-y plane, since the DOMs are populated more densly on the z axis
z_adjusted_metric = np.diag([1.0, 1.0, horizontal_dom_spacing / vertical_dom_spacing])

def get_coordinates(f, vertex_offsets, num_vertices, idx):
    """ Retrieves the coordinate matrix of an event. 
    
    Parameters:
    -----------
    f : hp5y.File
        The hd5 file.
    vertex_offsets : ndarray, shape [N]
        The offsets for each event.
    num_vertices : ndarray, shape [N]
        The number of vertices for each event.
    idx : int
        The index for which to create the coordinate matrix.
    
    Returns:
    --------
    C : ndarray, shape [n_vertices, 3]
        The coordinates for event idx.
    """
    offset = vertex_offsets[idx]
    length = num_vertices[idx]
    x = np.array(f['VertexX'][offset : offset + length])
    y = np.array(f['VertexY'][offset : offset + length])
    z = np.array(f['VertexZ'][offset : offset + length])
    return np.vstack([x, y, z]).T

def compute_adjacency_list(vertex_coordinates, k=50):
    """ Computes the adjacency list for a given coordiante matrix. 
    
    Parameters:
    -----------
    vertex_coordinates : ndarray, shape [n_events, 3]
        Coordinates for vertices of an event.
    k : int
        How many possible edges should be computed for each vertex. Use a high number, as this connectivity
        can be reduced (but not increased) during training time.
    
    Returns:
    --------
    adjacency_list : ndarray, shape [n_events, k]
        The indices of vertices a vertex is connected to, in increasing order, i.e. index 0 is closest, and index k-1 is most far.
    """
    N, _ = vertex_coordinates.shape
    graph = kneighbors_graph(
            vertex_coordinates, n_neighbors=min(N - 1, k), metric='mahalanobis', metric_params={'VI' : z_adjusted_metric}, mode='distance'
            ).todense()
    adjacency_list = np.argsort(-graph)[:, :k]
    return np.flip(np.array(adjacency_list, dtype=np.int16), axis=1)


if __name__ == '__main__':
    infile = sys.argv[1]
    k = int(sys.argv[2])
    with h5py.File(infile, 'r+') as f:
        num_vertices = np.array(f['NumberVertices'])
        vertex_offsets = num_vertices.cumsum() - num_vertices
        assert num_vertices.max() < 0x10000
        dset = f.create_dataset('AdjacencyList', (num_vertices.sum(), k), dtype='i2') # Use 16-bit ints
        for idx in tqdm(range(num_vertices.shape[0])):
            C = get_coordinates(f, vertex_offsets, num_vertices, idx)
            adjacency_list = compute_adjacency_list(C, k=k)
            dset[vertex_offsets[idx] : vertex_offsets[idx] + num_vertices[idx]] = adjacency_list

