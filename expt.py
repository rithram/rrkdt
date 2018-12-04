#! /usr/bin/env python

import argparse
import pandas as pd
import sys

from ground_truth_setup import preprocess_csv, preprocess_hdf5
from expt_methods import get_methods_for_expt
from nnsearch_utils import evaluate_setting
from plot_results import generate_figures as genfigs

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
      '-f',
      '--split_fraction',
      help='Reference/query split fraction of csv data',
      type=float,
      default=0.1
  )
  parser.add_argument(
      '-k', '--n_neighbors', help = 'Number of neighbors', type=int, default=10
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
      '-T', '--num_trees', help='The number of trees', type=int
  )

  args = parser.parse_args()
  return args
# -- end function


def main() :

    cmd_args = setup_cmd_args()

    print(cmd_args.data_type)
    assert cmd_args.data_type == 'csv' or cmd_args.data_type == 'hdf5'
    
    if cmd_args.data_type == 'csv' :
        assert (
            cmd_args.split_fraction > 0.0
            and cmd_args.split_fraction < 0.5
        ), (
            'Split fraction should be in (0.0, 0.5), %g provided'
            % cmd_args.split_fraction
        )

    assert cmd_args.n_neighbors > 0
    assert cmd_args.leaf_size > 0
    assert cmd_args.num_trees > 0

    assert cmd_args.results_file is not '', (
        'Please specify a results file via \'-r\' '
        'or \'--results_file\' to store results'
    )
    
    # Read in the data, set up references and queries and the true neighbors of
    # the queries
    data_pp_csv = lambda x : preprocess_csv(
        x, split_frac=cmd_args.split_fraction, k=cmd_args.n_neighbors
    )
    data_pp_hdf5 = lambda x : preprocess_hdf5(x, k=cmd_args.n_neighbors)
    data_pp = data_pp_csv if cmd_args.data_type == 'csv' else data_pp_hdf5
    references, queries, true_neighbors = data_pp(cmd_args.data)
    assert references.shape[1] == queries.shape[1]
    assert queries.shape[0] == true_neighbors.shape[0]
    assert true_neighbors.shape[1] == cmd_args.n_neighbors

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
        leaf_size=cmd_args.leaf_size, ntrees=cmd_args.num_trees
    )
    assert isinstance(all_methods, list)

    # 2. run experiment for each method
    all_method_results = []
    all_method_auprc = {}
    all_method_auprc['method'] = []
    all_method_auprc['auprc'] = []
    for method in all_methods :
        print('Processing %s ...' % method['name'])
        _, result_dfs, auprc_list = evaluate_setting(
            S=references,
            Q=queries,
            k=cmd_args.n_neighbors,
            indexer=method['indexer'],
            locater=method['locater'],
            hps=[ method['hparams'] ],
            true_list=true_neighbors
        )
        print('AUPRC for %s: %g' % (method['name'], auprc_list[0]))
        result_dfs[0]['method'] = method['name']
        all_method_results.append(result_dfs[0])
        all_method_auprc['method'].append(method['name'])
        all_method_auprc['auprc'].append(auprc_list[0])
        
    # 3. concatenate all results
    all_results = pd.concat(all_method_results)
    all_auprcs = pd.DataFrame.from_dict(all_method_auprc)
    #   a. output results to file
    #     i. Saving all results in a csv file
    rfile = cmd_args.results_file
    print('Saving results in \'%s\' ...' % rfile)
    all_results.to_csv(path_or_buf=rfile, header=True, index=False)
    #     ii. Saving the AUPRC for each method in a separate csv
    auprc_file = rfile + '.auprc'
    all_auprcs.to_csv(path_or_buf=auprc_file, header=True, index=False)

    # 4. Plot all results
    if cmd_args.figures_file is not '' :
        genfigs(all_results, all_method_auprc, cmd_args.figures_file)
    
    return 0
# -- end function


if __name__ == '__main__' :
#  try :
    status = main()
    sys.exit(status)
#  except Exception as e : 
#    print('Exception: %s (%s)' % (sys.exc_info()[0], e))
    #status = -1
# -- end function
