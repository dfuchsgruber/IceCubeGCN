import h5py
from glob import glob
import numpy as np
import sys
import random

# Use the 'DeltaLLH' column of the dataset to retrieve the number of events per datafile
PER_EVENT_COLUMN = 'NumberVertices'
SEED = 1337

random.seed(SEED)

def get_column(f, key):
    try: return f[key]['value']
    except: return f[key]['item']

if __name__ == '__main__':
    pattern = sys.argv[1]
    outfile = sys.argv[2]
    num_files = int(sys.argv[3])
    paths = list(glob(pattern))
    random.shuffle(paths)
    paths = paths[:num_files]
    print(f'Found {len(paths)} files that match the pattern.')
    #print(paths)
    
    
    dimensions = {}
    dtypes = {}
    # Calculate the dataset total size beforehand
    for idx, path in enumerate(paths):
        try:
            with h5py.File(path) as f:
                for key in f.keys():
                    if key == '__I3Index__': continue
                    if idx == 0:
                        dimensions[key] = 0
                        dtypes[key] = get_column(f, key).dtype
                    else:
                        assert key in dimensions, f'Dataset {path} contains key {key} which predecessor were missing.'
                        assert dtypes[key] == get_column(f, key).dtype, f'Different dtype {get_column(f, key)}'
                    dimensions[key] += get_column(f, key).shape[0]
        except Exception as e:
            print('\nFailed to parse file {path}.\n')
            raise e
        print(f'\rScanned file {idx} / {len(paths)}', end='\r')
    
    print(f'\nGot these final number of rows: {dimensions}')

    offsets = dict((key, 0) for key in dimensions)

    # Create output file
    with h5py.File(outfile, 'w') as outfile:
        # Create a dataset column for storing filenames
        outfile.create_dataset('filepath', (dimensions[PER_EVENT_COLUMN],), dtype=h5py.special_dtype(vlen=bytes))
        #print(outfile['filepath'])
        for key in dimensions:
            outfile.create_dataset(key, (dimensions[key],), dtype=dtypes[key])
        print(f'Created output file, filling now...')
        #print(paths)
        for path in paths:
            print(f'Exporting data from {path}', end='\n')
            with h5py.File(path) as src:
                if not PER_EVENT_COLUMN in src: continue # Some i3 files might not contain any events, skip those
                n_events = src[PER_EVENT_COLUMN].shape[0]
                outfile['filepath'][offsets[PER_EVENT_COLUMN] : offsets[PER_EVENT_COLUMN] + n_events] = bytes(path, encoding='ASCII')
                for key in dimensions:
                    print(f'\rCopying {key} from {path}...                             ', end='\r')
                    size = get_column(src, key).shape[0]
                    outfile[key][offsets[key] : offsets[key] + size] = get_column(src, key)
                    offsets[key] += size


