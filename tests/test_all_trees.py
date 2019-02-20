import sys
sys.path.insert(0, './src/')

from expt_methods import get_methods_for_expt as gmfe
from tests import test_tree as tt

import pandas as pd
import numpy as np

data = pd.read_csv('./tests/USPS.csv', header=None).values.astype(float)
print 'Data:', data.shape

methods = gmfe(5, 1)

for m in methods :
    print('Testing %s ...' % m['name'])
    indxr = lambda x,y : m['indexer'](x, y, log=False)
    lctr = m['locater']
    tt(data, indxr, lctr, m['hparams'])
