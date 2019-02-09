#! /usr/bin/env python

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import numpy as np
import pandas as pd
import sys
from timeit import default_timer
import pickle

from ground_truth_setup import preprocess_hdf5
from expt_methods import get_methods_for_expt

def setup_cmd_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='File containing data', type=str)
    parser.add_argument(
        '-M',
        '--methods',
        help='Comma-separated method(s) to evaluate',
        type=str,
        default='all'
    )
    parser.add_argument(
        '-r', '--results_file',
        help='File where the results would be output in csv format',
        type=str
    )
    parser.add_argument(
        '-g',
        '--figures_file',
        help='File where the figures will be output for this experiment',
        type=str,
        default=''
    )
    parser.add_argument(
        '-N', '--leaf_size', help='The minimum leaf size in the tree', type=int
    )
    parser.add_argument(
        '-R',
        '--n_reps',
        help='The number of times to evaluate a setting.',
        type=int,
        default=1
    )
    parser.add_argument(
        '-n', '--n_queries', help='Number of queries', type=int, default=10000
    )
  
    args = parser.parse_args()
    return args
# -- end function

def compute_musage(tree) :
    serialized_tree = pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)
    size1 = sys.getsizeof(serialized_tree)
    size2 = len(serialized_tree)
    print('GSO: %i, LEN: %i' % (size1, size2))

    return size1
# -- end function

def compute_indexing_time(S, indexer, hp) :
    start = default_timer()
    tree = indexer(S, hp)
    indexing_time = default_timer() - start

    return tree, indexing_time
# -- end function

def compute_query_time(tree, locater, Q) :
    start = default_timer()
    xx = [ locater(tree, q) for q in Q ]
    query_time = default_timer() - start

    return query_time
# -- end function

def compute_scaling_statistics(S, Q, indexer, hp, locater) :
    tree, itime = compute_indexing_time(S, indexer, hp)
    isize = compute_musage(tree)
    qtime = compute_query_time(tree, locater, Q)

    return itime, isize, qtime
# -- end function


def prep_queries(Q, nrows, ncols) :
    n, d = Q.shape
    assert d >= ncols
    ret = np.copy(Q)
    while n < nrows :
        ret = np.vstack((ret, Q))
        n, _ = ret.shape
    
    return ret[:nrows, :ncols]
# -- end function

def main() :

    cmd_args = setup_cmd_args()
    assert cmd_args.leaf_size > 0
    assert cmd_args.n_reps > 0
    assert cmd_args.n_queries > 0

    # 1. get all methods
    all_methods = get_methods_for_expt(
        leaf_size=cmd_args.leaf_size, ntrees=1
    )
    assert isinstance(all_methods, list)

    methods_list = [ method['name'] for method in all_methods ]

    methods_input = 'all' if cmd_args.methods is 'all' else []
    if cmd_args.methods is not 'all' :
        mlist = cmd_args.methods.split(',')
        for m in mlist :
            assert m in methods_list, (
                'Invalid method %s specified, must be one of \n%s'
                % (m, str(methods_list))
            )
            methods_input.append(m)
    
    assert cmd_args.results_file is not '', (
        'Please specify a results file via \'-r\' '
        'or \'--results_file\' to store results'
    )
    
    # Read in the data, set up references and queries
    # (no need for true neighbors here)
    references, queries, _ = preprocess_hdf5(cmd_args.data, k=0)
    nrows, ncols = references.shape
    assert ncols == queries.shape[1], (
        'References have %i columns, queries have %i'
        % (ncols, queries.shape[1])
    )

    # Pick number of columns
    ncols2 = 2
    while ncols2 * 2 <= ncols :
        ncols2 *= 2
    print('Performing experiment on %i columns' % ncols2)
    references = references[:, :ncols2]
    print 'S: ', references.shape

    # process queries
    queries = prep_queries(queries, cmd_args.n_queries, ncols2)
    print 'Q: ', queries.shape

    ncols = ncols2

    # 2. Choose values for nrows scaling
    # 3. For each nrows, create nreps seeds for subsampling reference set
    # 4. For each nrows
    #   a. For each method
    #     i . Select reference subset using preselected seed
    #     ii. Compute statistics for current subset
    #   b. Average stats across repetitions
    #   c. Save 'nrows,method,indexing_time(mean,std),index_size(mean,std),query_time'
    # 5. Save results to file
    # 6. Plot results for n vs. all stats


    # 2. Choose values for nrows scaling
    n = 1024
    assert n <= nrows, ('#rows: %i' % nrows)
    n_list = []
    while n < nrows :
        n_list.append(n)
        n *= 2
    n_list.append(nrows)
    print('Trying the following value of n for scaling:\n%s' % repr(n_list))

    # 3. For each nrows, create nreps permutations for subsampling reference set
    rep_perms = [ np.random.permutation(range(nrows)).tolist() for i in range(cmd_args.n_reps) ]
    
    n_scaling_results = {
        'n' : [],
        'method' : [],
        'mean_indexing_time' : [],
        'std_indexing_time' : [],
        'mean_index_size' : [],
        'std_index_size' : [],
        'mean_query_time' : [],
        'std_query_time' : [],
    }        
    # 4. For each nrows
    for n in n_list :
        print('Processing n=%i' % n)
        # a. For each method
        for method in all_methods :
            if not cmd_args.methods == 'all' and not method['name'] in methods_input :
                print('Skipping %s since %s selected' % (method['name'], str(methods_input)))
                continue
            # i  . For each repetition
            rep_results = { 'itime' : [], 'isize': [], 'qtime' : [] }
            for i in range(cmd_args.n_reps) :
                # I . Select reference subset using preselected permutation
                rep_perm = rep_perms[i]
                S = references[rep_perm[:n], :]
                nr, nc = S.shape
                print('Reference subsample: %i x %i' % (nr, nc))
                assert nr == n and nc == ncols
                # II. Compute statistics for current subset
                itime, isize, qtime = compute_scaling_statistics(
                    S, queries, method['indexer'], method['hparams'], method['locater']
                )
                rep_results['itime'].append(itime)
                rep_results['isize'].append(isize)
                rep_results['qtime'].append(qtime)
            # ii . Average stats across repetitions
            mean_itime = np.mean(rep_results['itime'])
            std_itime = np.std(rep_results['itime'])
            mean_isize = np.mean(rep_results['isize'])
            std_isize = np.std(rep_results['isize'])
            mean_qtime = np.mean(rep_results['qtime'])
            std_qtime = np.std(rep_results['qtime'])
            
            # iii. Save 'nrows,method,indexing_time(mean,std),index_size(mean,std),query_time(mean,std)'
            print(
                '%s ==> n: %i, itime: %g +- %g, isize: %g +- %g, qtime: %g +- %g'
                % (
                    method['name'], n, mean_itime, std_itime, mean_isize, std_isize, mean_qtime, std_qtime
                )
            ) 
            n_scaling_results['n'].append(n)
            n_scaling_results['method'].append(method['name'])
            n_scaling_results['mean_indexing_time'].append(mean_itime)
            n_scaling_results['std_indexing_time'].append(std_itime)
            n_scaling_results['mean_index_size'].append(mean_isize)
            n_scaling_results['std_index_size'].append(std_isize)
            n_scaling_results['mean_query_time'].append(mean_qtime)
            n_scaling_results['std_query_time'].append(std_qtime)

    # 5. Save results to file
    rfile = cmd_args.results_file + '.nscaling.csv'
    print('Saving n-scaling results in \'%s\' ...' % rfile)
    all_results = pd.DataFrame.from_dict(n_scaling_results)
    all_results.to_csv(path_or_buf=rfile, header=True, index=False)

    # 6. Plot results for n vs. all stats
    if cmd_args.figures_file is not '' :
        raise NotImplementedError()
    
    return 0
# -- end function

if __name__ == '__main__' :
    status = main()
    sys.exit(status)
# -- end function
