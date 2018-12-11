import argparse
import sys

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from nnsearch_utils import compute_true_nn


# Function to read in a csv file, split it into references
# and queries and compute the true nearest-neighbors.
def preprocess_csv(
    filename, split_frac=0.1, k=10, header=None, cols2drop=None
) :
    # Read in data
    data = pd.read_csv(filename, header=header, engine='python')
    nrows, ncols = data.shape
    print('Data (%i x %i)' % (nrows, ncols))
    if cols2drop is not None :
        assert type(cols2drop) is list
        data.drop(cols2drop, axis=1)

    # Split dataset into references and queries
    R, Q = train_test_split(data, test_size=split_frac, random_state=5489)
    nrefs, _ = R.shape
    nqurs, _ = Q.shape
    print('Data split into %i-%i' % (nrefs, nqurs))

    # Compute true nearest neighbors
    print('Computing true nearest-neighbors ...')
    true_nns = compute_true_nn(R.values, Q.values, k=k)

    return R.values.astype(float), Q.values.astype(float), true_nns
# -- end function

# Function to read in a hdf5 file, pull out the references
# and query files, and the true nearest-neighbors
def preprocess_hdf5(filename, k=10) :
    data_f = h5py.File(filename)

    nqueries, knn_available = data_f['neighbors'].shape
    assert k <= knn_available, (
        '%i NN available, %i NN requested ...'
        % (knn_available, k)
    )

    # Get the references and queries
    R = np.array(data_f['train'])
    print('References (%i x %i)' % (R.shape[0], R.shape[1]))

    Q = np.array(data_f['test'])
    print('Queries (%i x %i)' % (Q.shape[0], Q.shape[1]))

    # Extract the neighbors
    true_nns = np.array(data_f['neighbors'])[:,0:k]

    return R, Q, true_nns
# -- end function


# Function to read in csv, generate neighbors and save in an HDF5 for reuse
# Mainly useful for large datasets
def csv2hdf5(out_hdf5_file, in_csv_file_R, k=100, in_csv_file_Q=None, Q_size=10000) :
    # Read in data
    data = pd.read_csv(in_csv_file_R, header=None, engine='python')
    nrows, ncols = data.shape
    print('Data (%i x %i)' % (nrows, ncols))

    # Split dataset into references and queries
    R, Q = None, None
    nrows_R, nrows_Q = 0, 0
    if in_csv_file_Q is None :
        assert Q_size > 0
        R, Q = train_test_split(data, test_size=Q_size, random_state=5489)
        nrows_R, _ = R.shape
        nrows_Q, _ = Q.shape
        print('Data split into %i-%i' % (nrows_R, nrows_Q))
    else :
        Q = pd.read_csv(in_csv_file_Q, header=None, engine='python')
        nrows_Q, ncols_Q = Q.shape
        assert ncols_Q == ncols, ('R dimension %i, Q dimension %i' % (ncols, ncols_Q))
        R = data
        nrows_R = nrows
        print('Data input as R(%i), Q(%i) points' % (nrows, nrows_Q))

    # Compute true nearest neighbors
    print('Computing true nearest-neighbors ...')
    true_nns = compute_true_nn(R.values, Q.values, k=k)

    # Open HDF5 file and write 'train', 'test', 'neighbors'
    print('Writing results into HDF5 file ...')
    outf = h5py.File(out_hdf5_file, 'w')
    outf.create_dataset('train', (nrows_R, ncols), dtype=R.values.dtype)[:] = R.values
    outf.create_dataset('test', (nrows_Q, ncols), dtype=Q.values.dtype)[:] = Q.values
    outf.create_dataset('neighbors', (nrows_Q, k), dtype='i')[:] = true_nns
    outf.close()

    return
# -- end function

def csv2hdf5_cmdline() :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--hdf5_out', help='HDF5 file to be output', type=str
    )
    parser.add_argument('-i', '--csv_in', help='CSV file input', type=str)
    parser.add_argument(
        '-k', '--n_neighbors', help='Number of neighbors', type=int
    )
    parser.add_argument(
        '-q',
        '--queries',
        help='CSV file for queries if available',
        type=str,
        default=''
    )
    parser.add_argument(
        '-n',
        '--n_queries',
        help='Number of points to set aside as queries',
        type=int,
        default=0
    )

    args = parser.parse_args()

    qfile = None if args.queries == '' else args.queries
    csv2hdf5(
        out_hdf5_file=args.hdf5_out,
        in_csv_file_R=args.csv_in,
        k=args.n_neighbors,
        in_csv_file_Q=qfile,
        Q_size=args.n_queries
    )
# -- end function

if __name__ == '__main__' :
    status = csv2hdf5_cmdline()
    sys.exit(status)
