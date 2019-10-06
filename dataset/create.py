from icecube import icetray, dataclasses, dataio, phys_services
from I3Tray import I3Tray
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import sys
import tables
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from label import classify_wrapper

# Parses i3 files in order to create a (huge) hdf5 file that contains all events of all files

with open('dom_positions.pkl', 'rb') as f:
    dom_positions = pickle.load(f)

# Scale all coordinates by this amount, those values are empirical
coordinate_scale = [37.691566, 36.396854, 41.384834]

# Feature columns per vertex of a graph in the output hdf5 file
vertex_features = [
    'ChargeFirstPulse', 'ChargeLastPulse', 'ChargeMaxPulse', 'TimeFirstPulse', 'TimeLastPulse', 'TimeMaxPulse',
    'TotalCharge', 'TimeStd',
]

def normalize_coordinates(coordinates, weights=None, scale=coordinate_scale, copy=False):
    """ Normalizes the coordinates matrix by centering it using weighs (like charge).
    
    Parameters:
    -----------
    coordinates : ndarray, shape [N, 3]
        The coordinates matrix to be centered.
    weights : ndarray, shape [N] or None
        The weights for each coordinates.
    scale : ndarray, shape [3] or None
        Scale to apply to each coordinate. If None is given, the standard deviation along each axis is used.
    copy : bool
        If true, the coordinate matrix will be copied and not overwritten.
        
    Returns:
    --------
    coordinates : ndarray, shape [N, 3]
        The normalized coordinate matrix.
    mean : ndarray, shape [3]
        The means along all axis
    scale : ndarray, shape [3]
        The scaling along each axis.
    """
    if copy:
        coordinates = coordinates.copy()
    mean = np.average(coordinates, axis=0, weights=weights).reshape((1, -1))
    coordinates -= mean
    if scale is None:
        scale = np.sqrt(np.average(coordinates ** 2, axis=0, weights=None)) + 1e-20
    coordinates /= scale
    return coordinates, mean.flatten(), scale


def get_events_from_frame(frame, charge_threshold=0.5, time_scale=1.0, charge_scale=1.0, pulses_key='InIceDSTPulses'):
    """ Extracts data (charge and time of first arrival) as well as their positions from a frame.
    
    Parameters:
    -----------
    frame : ?
        The physics frame to extract data from.
    charge_threshold : float
        The threshold for a charge to be considered a valid pulse for the time of the first pulse arrival (before scaling).
    charge_scale : float
        The normalization constant for charge.
    time_scale : float
        The normalization constant for time.
    pulses_key : str
        The key for the pulse series in the I3 file.
    
    Returns:
    --------
    dom_features : dict
        A dict from features to ndarrays of shape [N,] representing the features for each vertex.
    vertices : dict
        A dict from coordiante axis to ndarrays of shape [N,] representing the positions for each vertex on an axis.
        Coordinate matrix for the doms that were active during this event.
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    x = frame[pulses_key]
    hits = x.apply(frame)

    features = {feature : [] for feature in vertex_features}
    vertices = {axis : [] for axis in ('x', 'y', 'z')}

    # For each event compute event features based on the pulses
    omkeys = []
    for omkey, pulses in hits:
        omkey = tuple(omkey) # The pickle maps triplets to positions
        charges, times = [], []
        for pulse in pulses:
            if pulse.charge >= charge_threshold:
                charges.append(pulse.charge)
                times.append(pulse.time + frame['TimeShift'].value)

        charges = np.array(charges) * charge_scale
        if charges.shape[0] == 0: continue # Don't use DOMs that recorded no charge above the threshold
        times = np.array(times)
        times = times * time_scale
        # Calculate the charge weighted standard devation of times
        charge_weighted_mean_time = np.average(times, weights=charges)
        charge_weighted_std_time = np.sqrt(np.average((times - charge_weighted_mean_time)**2, weights=charges))

        features['ChargeFirstPulse'].append(charges[0])
        features['ChargeLastPulse'].append(charges[-1])
        features['ChargeMaxPulse'].append(charges.max())
        features['TimeFirstPulse'].append(times[0])
        features['TimeLastPulse'].append(times[-1])
        features['TimeMaxPulse'].append(times[charges.argmax()])
        features['TimeStd'].append(charge_weighted_std_time)
        features['TotalCharge'].append(charges.sum())
        for axis in vertices:
            vertices[axis].append(dom_positions[omkey][axis])
        assert omkey not in omkeys
        omkeys.append(omkey)

    return features, vertices, np.array(omkeys)

def process_frame(frame, charge_scale=1.0, time_scale=1e-3):
    """ Processes a frame to create an event graph and metadata out of it.
    
    Parameters:
    -----------
    frame : ?
        The data frame to extract an event graph with features from.
    charge_scale : float
        The normalization constant for charge.
    time_scale : float
        The normalization constant for time.
    """

    ### Meta data of the event for analysis of the classifier and creation of ground truth
    nu = dataclasses.get_most_energetic_neutrino(frame['I3MCTree'])

    # Obtain the PDG Encoding for ground truth
    frame['PDGEncoding'] = dataclasses.I3Double(nu.pdg_encoding)
    frame['InteractionType'] = dataclasses.I3Double(frame['I3MCWeightDict']['InteractionType'])
    frame['NeutrinoEnergy'] = dataclasses.I3Double(frame['trueNeutrino'].energy)
    # Some rare events do not produce a cascade
    try:
        frame['CascadeEnergy'] = dataclasses.I3Double(frame['trueCascade'].energy)
    except:
        frame['CascadeEnergy'] = dataclasses.I3Double(np.nan)
    try:
        # Appearently also frames with no primary muon contain this field, so to distinguish try to access it (which should throw an exception)
        frame['MuonEnergy'] = dataclasses.I3Double(frame['trueMuon'].energy)
        frame['TrackLength'] = dataclasses.I3Double(frame['trueMuon'].length)
    except:
        frame['MuonEnergy'] = dataclasses.I3Double(np.nan)
        frame['TrackLength'] = dataclasses.I3Double(np.nan)
    frame['RunID'] = icetray.I3Int(frame['I3EventHeader'].run_id)
    frame['EventID'] = icetray.I3Int(frame['I3EventHeader'].event_id)
    frame['PrimaryEnergy'] = dataclasses.I3Double(nu.energy)

    ### Create features for each event 
    features, coordinates, _ = get_events_from_frame(frame, charge_scale=charge_scale, time_scale=time_scale)
    for feature_name in vertex_features:
        frame[feature_name] = dataclasses.I3VectorFloat(features[feature_name])
    
    ### Create offset lookups for the flattened feature arrays per event
    frame['NumberVertices'] = icetray.I3Int(len(features[features.keys()[0]]))

    ### Create coordinates for each vertex
    C = np.vstack(coordinates.values()).T
    C, C_mean, C_std = normalize_coordinates(C, weights=None, copy=True)
    C_cog, C_mean_cog, C_std_cog = normalize_coordinates(C, weights=features['TotalCharge'], copy=True)


    frame['VertexX'] = dataclasses.I3VectorFloat(C[:, 0])
    frame['VertexY'] = dataclasses.I3VectorFloat(C[:, 1])
    frame['VertexZ'] = dataclasses.I3VectorFloat(C[:, 2])
    frame['COGCenteredVertexX'] = dataclasses.I3VectorFloat(C_cog[:, 0])
    frame['COGCenteredVertexY'] = dataclasses.I3VectorFloat(C_cog[:, 1])
    frame['COGCenteredVertexZ'] = dataclasses.I3VectorFloat(C_cog[:, 2])

    ### Output centering and true debug information
    frame['PrimaryXOriginal'] = dataclasses.I3Double(nu.pos.x)
    frame['PrimaryYOriginal'] = dataclasses.I3Double(nu.pos.y)
    frame['PrimaryZOriginal'] = dataclasses.I3Double(nu.pos.z)
    frame['CMeans'] = dataclasses.I3VectorFloat(C_mean)
    frame['COGCenteredCMeans'] = dataclasses.I3VectorFloat(C_mean_cog)

    ### Compute targets for possible auxilliary tasks, i.e. position and direction of the interaction
    frame['PrimaryX'] = dataclasses.I3Double((nu.pos.x - C_mean[0]) / C_std[0])
    frame['PrimaryY'] = dataclasses.I3Double((nu.pos.y - C_mean[1]) / C_std[1])
    frame['PrimaryZ'] = dataclasses.I3Double((nu.pos.z - C_mean[2]) / C_std[2])

    frame['COGCenteredPrimaryX'] = dataclasses.I3Double((nu.pos.x - C_mean_cog[0]) / C_std_cog[0])
    frame['COGCenteredPrimaryY'] = dataclasses.I3Double((nu.pos.y - C_mean_cog[1]) / C_std_cog[1])
    frame['COGCenteredPrimaryZ'] = dataclasses.I3Double((nu.pos.z - C_mean_cog[2]) / C_std_cog[2])
    frame['PrimaryAzimuth'] = dataclasses.I3Double(nu.dir.azimuth)
    frame['PrimaryZenith'] = dataclasses.I3Double(nu.dir.zenith)

    ### Compute possible reco inputs that apply to entire event sets
    track_reco = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track']
    frame['RecoX'] = dataclasses.I3Double((track_reco.pos.x - C_mean[0]) / C_std[0])
    frame['RecoY'] = dataclasses.I3Double((track_reco.pos.y - C_mean[1]) / C_std[1])
    frame['RecoZ'] = dataclasses.I3Double((track_reco.pos.z - C_mean[2]) / C_std[2])
    frame['COGCenteredRecoX'] = dataclasses.I3Double((track_reco.pos.x - C_mean_cog[0]) / C_std_cog[0])
    frame['COGCenteredRecoY'] = dataclasses.I3Double((track_reco.pos.y - C_mean_cog[1]) / C_std_cog[1])
    frame['COGCenteredRecoZ'] = dataclasses.I3Double((track_reco.pos.z - C_mean_cog[2]) / C_std_cog[2])
    frame['RecoAzimuth'] = dataclasses.I3Double(track_reco.dir.azimuth)
    frame['RecoZenith'] = dataclasses.I3Double(track_reco.dir.zenith)

    ### Apply labeling
    classify_wrapper(p_frame, None, gcdfile=None)
    return True


def create_dataset(outfile, infiles):
    """
    Creates a dataset in hdf5 format.

    Parameters:
    -----------
    outfile : str
        Path to the hdf5 file.
    paths : dict
        A list of intput i3 files.
    """
    infiles = infiles
    tray = I3Tray()
    tray.AddModule('I3Reader',
                FilenameList = infiles)
    tray.AddModule(process_frame, 'process_frame')
    tray.AddModule(I3TableWriter, 'I3TableWriter', keys = vertex_features + [
        # Meta data
        'PDGEncoding', 'InteractionType', 'NeutrinoEnergy', 
        'CascadeEnergy', 'MuonEnergy', 'TrackLength'
        'RunID', 'EventID',
        # Lookups
        'NumberVertices',
        # Coordinates and pairwise distances
        'VertexX', 'VertexY', 'VertexZ', 
        'COGCenteredVertexX', 'COGCenteredVertexY', 'COGCenteredVertexZ',
        # Auxilliary targets
        'PrimaryX', 'PrimaryY', 'PrimaryZ', 
        'COGCenteredPrimaryX', 'COGCenteredPrimaryY', 'COGCenteredPrimaryZ', 
        'PrimaryAzimuth', 'PrimaryZenith', 'PrimaryEnergy',
        # Debug stuff
        'PrimaryXOriginal', 'PrimaryYOriginal', 'PrimaryZOriginal',
        'CMeans', 'COGCenteredCMeans',
        # Class label
        'classification',
        ], 
                TableService=I3HDFTableService(outfile),
                SubEventStreams=['InIceSplit'],
                BookEverything=False
                )
    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    create_dataset(sys.argv[1], sys.argv[2])
