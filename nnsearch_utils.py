import timeit

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import auc

#import matplotlib.pyplot as plt
#import matplotlib.lines as mlines

# Basic function to evaluate the recall of the search
def recall(true_set, candidate_set) :
    nfound = true_set & candidate_set
    return float(len(nfound)) / len(true_set)

# Basic function to evaluate the precision of the search
def precision(true_set, candidate_set) :
    nfound = true_set & candidate_set
    return float(len(nfound)) / len(candidate_set)

# Generate a precision-recall using the
#  results from the search with RP-tree forest
def generate_results(true_list, candidate_lists, query_time, indexing_time=None) :
    '''
    Generate the average precision, recall and query times 
    incrementally for each tree.

    Parameters
    ----------
    true_list: list [ list [ indices ] ] ( # queries x k )
      For each query, the list of neighbors.

    candidate_lists: list [ list [ list [ indices ] ] ]  ( # queries x # trees x ~leaf size )
      For each query, for each tree, the list of candidates obtained.

    query_time: list [ float ] ( # trees )
      For all queries, the query time to process a single tree.

    indexing_time: list [ float ] ( # trees )
      The time to build each of the single trees.


    Output
    ------
    Dataframe with 3 columns: average recall, average precision, (query time
      The number of rows of the data frame would be # trees.
      All the metrics would be cumulative as we go down the rows.
    
    '''

    assert len(true_list) == len(candidate_lists)
    nqueries = len(true_list)
    
    k = len(true_list[0])
    for l in true_list :
        assert len(l) == k

    ntrees = len(candidate_lists[0])
    for l in candidate_lists :
        assert len(l) == ntrees
    assert ntrees == len(query_time)

    recall_sum = np.zeros(ntrees)
    precision_sum = np.zeros(ntrees)


    for tlist, clist in zip(true_list, candidate_lists) :
        tset = set(tlist)
        cset = set([])

        for tree_index, cl in enumerate(clist) :
            cset |= set(cl)
            recall_sum[tree_index] += recall(tset, cset)
            precision_sum[tree_index] += precision(tset, cset)

    avg_recall = recall_sum / nqueries
    avg_precision = precision_sum / nqueries
    cum_query_time = np.cumsum(np.array(query_time))

    result_dict = {
        'recall' : avg_recall.tolist(),
        'precision' : avg_precision.tolist(),
        'query_time' : cum_query_time.tolist()
    }

    if indexing_time is not None :
        result_dict['indexing_time'] = np.cumsum(np.array(indexing_time)).tolist()

    result_df = pd.DataFrame.from_dict(result_dict)

    return result_df

def ComputeAUPRC(results) :
    precision = results['precision'].values
    recall = results['recall'].values
    assert len(precision) == len(recall)

    x_auc = np.concatenate([ [ 0.0 ], recall, [ recall[-1] ] ])
    y_auc = np.concatenate([ [ precision[0] ], precision, [ 0.0 ] ])

    return auc(x=x_auc, y=y_auc)


# Generate the true answers using sklearn's NNS
def compute_true_nn(S, Q, k) :
    nns = NN(n_neighbors=k, algorithm='auto', metric='l2')
    nns.fit(S)
    true_list = nns.kneighbors(Q, return_distance=False)
    return true_list


def evaluate_setting(
        S,
        Q,
        k,
        indexer=None,
        locater=None,
        hps=[],
        true_list=None
) :
    if len(hps) == 0 :
        raise Exception('No parameters to evaluate, terminating!')

    for hp in hps :
        if hp.leaf_size < k :
            raise Exception(
                'Leaf size %i, while k is %i ... '
                'Recall will probably never reach 1'
                % (hp.leaf_size, k)
            )

    if indexer is None or locater is None :
        raise Exception(
            'ERROR: Both preprocessing function and querying '
            'function needs to be specified'
        )
        return

    if true_list is None :
        # If true neighbor set is not provided, compute it
        print('Computing the true sets of size %i' % k)
        true_list = compute_true_nn_sets(S, Q, k)

    # The return variable is a map indexed on k since
    #  the final single plot would contain all hps for a
    #  single k

    result_dfs = []
    auprc_list = []
    
    for hp in hps :
        print(
            'Pre-processing set into %i trees with leaf-size %i '
            'using indexer \'%s\' ...'
            % (hp.ntrees, hp.leaf_size, indexer.__name__)
        )
        print('Performing the search on %i queries with locater \'%s\' ...'
              % (len(Q), locater.__name__)
        )

        #trees = [ indexer(S, hp) for i in range(hp.ntrees) ]
 

        query_time = []
        indexing_time = []
        candidate_lists = [ [] for i in range(len(Q)) ]

        for i in range(hp.ntrees) :

            # Build the tree
            start = timeit.default_timer()
            tree = indexer(S, hp)
            stop = timeit.default_timer()

            indexing_time.append(stop - start)

            # Query the tree
            start = timeit.default_timer()
            all_q_results = [ locater(tree, q) for q in Q ]
            stop = timeit.default_timer()

            query_time.append(stop - start)

            # Store results
            for i in range(len(Q)) :
                candidate_lists[i].append(all_q_results[i])

        results = generate_results(
            true_list, candidate_lists, query_time, indexing_time
        )
        results['leaf_size'] = hp.leaf_size
        auprc = ComputeAUPRC(results)

        result_dfs.append(results)
        auprc_list.append(auprc)

    return true_list, result_dfs, auprc_list

