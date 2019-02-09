#! /usr/bin/env python

#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import numpy as np
import pandas as pd
import sys
import pickle

from ground_truth_setup import preprocess_csv, preprocess_hdf5
from expt_methods import get_methods_for_expt

def setup_cmd_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='File containing data', type=str)
    parser.add_argument(
        '-t',
        '--data_type',
        help='Data file type -- should be one of \'csv\' or \'hdf5\'',
        type=str
    )
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
    #parser.add_argument(
    #    '-T', '--num_trees', help='The number of trees', type=int
    #)
    parser.add_argument(
        '-R',
        '--n_reps',
        help='The number of times to evaluate a setting.',
        type=int,
        default=1
    )
  
    args = parser.parse_args()
    return args
# -- end function

def compute_musage(S, indexer, hp) :
    tree = indexer(S, hp)
    serialized_tree = pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)
    size1 = sys.getsizeof(serialized_tree)
    size2 = len(serialized_tree)
    print('GSO: %i, LEN: %i' % (size1, size2))

    return size1
# -- end function


def main() :

    cmd_args = setup_cmd_args()

    assert cmd_args.data_type == 'csv' or cmd_args.data_type == 'hdf5'
    assert cmd_args.leaf_size > 0
    #assert cmd_args.num_trees > 0
    assert cmd_args.n_reps > 0

    methods_list = [
        'RPTree',
        'SpGa:RPT(1/10)', 'SpGa:RPT(1/3)',
        'SpRa:RPT(1/10)', 'SpRa:RPT(1/3)',
        'RR:KDTree', 'RC:KDTree', 'FF:KDTree'
    ]

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
    
    # Read in the data, set up references and queries and the true neighbors of
    # the queries
    data_pp_csv = lambda x : preprocess_csv(
        x, split_frac=0.01, k=1
    )
    data_pp_hdf5 = lambda x : preprocess_hdf5(x, k=1)
    data_pp = data_pp_csv if cmd_args.data_type == 'csv' else data_pp_hdf5
    references, _, _ = data_pp(cmd_args.data)
    # Run the experiments for each method
    # 1. get all methods
    # 2. run experiment for each method
    # 3. concatenate all results
    # 4. Plot all results
    #   a. Plot PR-curve for each method (line plots) 
    #   b. Plot AUPRC for each method (bar plots)
    #   c. #queries/second curve for each method (line plots)

    # 1. get all methods
    all_methods = get_methods_for_expt(
        leaf_size=cmd_args.leaf_size, ntrees=1
    )
    assert isinstance(all_methods, list)

    # 2. run experiment for each method
    all_method_res = {}
    all_method_res['method'] = []
    all_method_res['mean'] = []
    all_method_res['std'] = []
    for method in all_methods :
        if not cmd_args.methods == 'all' and not method['name'] in methods_input :
            print('Skipping %s since %s selected' % (method['name'], str(methods_input)))
            continue
        
        print('Processing %s ...' % method['name'])
        res_list = []
        for i in range(cmd_args.n_reps) :
            res_list.append(compute_musage(
                S=references,
                indexer=method['indexer'],
                hp=method['hparams']
            ))
        all_method_res['method'].append(method['name'])
        all_method_res['mean'].append(np.mean(res_list))
        all_method_res['std'].append(np.std(res_list))
        print(
            'Memory usage for %s: %g %s'
            % (method['name'], np.mean(res_list), str(res_list))
        )

    rfile = cmd_args.results_file
    print('Saving results in \'%s\' ...' % rfile)
    all_results = pd.DataFrame.from_dict(all_method_res)
    all_results.to_csv(path_or_buf=rfile, header=True, index=False)

    # 4. Plot all results
    if cmd_args.figures_file is not '' :
        raise NotImplementedError()
    
    return 0
# -- end function


if __name__ == '__main__' :
    status = main()
    sys.exit(status)
# -- end function
