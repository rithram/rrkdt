import pandas as pd
import numpy as np

def test_tree(S, indexer, locater, hparams) :
    for i in range(10) :
        missed_idxs = []
        tree = indexer(S, hparams)
        print(tree.keys())
        for idx, q in enumerate(S) :
            cset = locater(tree, q)
            if idx not in cset :
                print('Index %i, C.set:%s' % (idx, str(cset)))
                missed_idxs.append(idx)
        print('Missed %i/%i points' % (len(missed_idxs), len(S)))
        print(missed_idxs)
# -- end function

from ff_kdtree import build_ff_kdtree, search_ff_kdtree

data = pd.read_csv('/scratch/datasets/DataSet/USPS.csv', header=None)
print('Data:', data.shape)

class HParams :
    leaf_size = None
    ntrees = None
# -- end class

srp_hparam = HParams()
srp_hparam.leaf_size = 5
srp_hparam.ntrees = 1
#srp_hparam.use_sign = False
#srp_hparam.col_prob = 0.1
#print('Testing SpGa:RPTree ...')
#indxr = lambda x,y : build_sparse_rptree(x, y, log=False)
#lctr = search_sparse_rptree
print('Testing FF:KDTree ...')
indxr = lambda x,y : build_ff_kdtree(x, y, log=False)
lctr = search_ff_kdtree
test_tree(data.values.astype(float), indxr, lctr, srp_hparam)

#srp_hparam.use_sign = True
#print('Testing SpRa:RPTree ...')
#test_tree(data.values.astype(float), indxr, lctr, srp_hparam)
