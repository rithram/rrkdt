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
    data = pd.read_csv(filename, header)
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

    return R.values, Q.values, true_nns
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
