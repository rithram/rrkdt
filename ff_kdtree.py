import numpy as np
from numpy.random import permutation as rperm
from collections import deque

from rptree import rplog
from rptree import Node
from rptree import split_node

from ffht import fht

def build_ff_kdtree(S, hparams, log=False) :
    logr = lambda message : rplog(message, log)
    
    nrows, ncols = S.shape
    leaf_size = hparams.leaf_size
    logr(
        'Building k-d tree with data pre-conditioned with FastFood '
        'on %i points in %i dims; max. leaf size: %i' 
        % (nrows, ncols, leaf_size)
    )

    # Generate a random diagonal sign matrix D
    diag_sign = np.random.binomial(n=1, p=0.5, size=ncols) * 2 - 1
    # D x
    D_S = np.multiply(S, diag_sign)

    # Pad each point to have some power of 2 size
    lncols = np.log2(ncols)
    new_ncols = ncols if int(lncols) == lncols else np.power(2, int(lncols) + 1)
    logr('Padding %i features to %i with 0' % (ncols, new_ncols))
    pad_vec = np.zeros(new_ncols - ncols)

    dense_x_list = []
    for i in range(nrows) :
        x = np.concatenate([ D_S[i], pad_vec ])
        # H D x
        fht(x)
        dense_x_list.append(x)

    HD_S = np.array(dense_x_list)
#    HD_S /= np.sqrt(float(new_ncols))

    # Permute each point in a random (but repeated manner)
    # Multiply with a random gaussian diagonal matrix
    # Perform second WHT

    # Generate a random diagonal gaussian matrix G
    diag_gaussian = np.random.normal(size=new_ncols)
    perm_seed = np.random.randint(99999)

    dense_x_list = []
    for p in HD_S :
        # G \Pi H D x
        # setting the seed here to ensure that the permutation
        # is same across all points
        np.random.seed(perm_seed)
        x2 = diag_gaussian * rperm(p)

        # H G \Pi H D x
        fht(x2)
        dense_x_list.append(x2)

    HGPHD_S = np.array(dense_x_list)
 #   HGPHD_S /= np.sqrt(float(new_ncols))
    HGPHD_S /= float(new_ncols)
    
    logr('FastFood-ed data matrix has shape %s previously %s'
         % (repr(HGPHD_S.shape), repr(S.shape))
    )
    
    a, b = HGPHD_S.shape
    assert a == nrows and b == new_ncols
    
    nodes = deque()
    nidx = 0
    root = Node(0, 0)
    nodes.append((root, root.level, range(nrows)))
    
    while len(nodes) > 0 :
        n, l, idxs = nodes.popleft()
        indent = str('|-') * l;
        logr(
            '%sLevel %i, node %i, %i points ....' 
            % (indent, l, n.idx, len(idxs))
        )
        # choose column equaling level % new_ncols
        colidx = l % new_ncols
            
        nidx = split_node(
            hparams.leaf_size,
            HGPHD_S[:, colidx],
            idxs,
            indent,
            n,
            nidx,
            l + 1,
            nodes,
            logr
        )
    
    return {
        'tree' : root,
        'pad' : pad_vec,
        'D' : diag_sign,
        'ncols' : ncols,
        'new_ncols' : new_ncols,
        'P_seed' : perm_seed,
        'G' : diag_gaussian
    }
# -- end function

def traverse_ff_kdtree(tree, log=False) :
    logr = lambda message : rplog(message, log)
    nodes = deque()
    nodes.append(tree['tree'])
    print('D:', tree['D'])
    print('G:', tree['G'])
    print('Seed for random P:', tree['P_seed'])
    ncols = tree['ncols']
    new_ncols = tree['new_ncols']

    print('Data dimensionality %i --> %i' % (ncols, new_ncols))
    
    while len(nodes) > 0 :
        n = nodes.popleft()
        l = n.level
        indent = str('|-') * l
        ms = ''
        if n.leaf :
            ms = 'pidxs:' + str(n.pidxs)
        else :
            nodes.append(n.lchild)
            nodes.append(n.rchild)
            ms = 'New col:' + str(n.level % new_ncols) + ', val:' + str(n.val)
        logr(
            '%sL %i: leaf?%i, id:%i --> %s' 
            % (indent, l, n.leaf, n.idx, ms)
        )
# -- end function

def HGPHD_x(tree, q) :
    # HD_x
    HD_x = np.concatenate([ q * tree['D'], tree['pad'] ])
    fht(HD_x)

    # GPHD_x
    np.random.seed(tree['P_seed'])
    HGPHD_x = tree['G'] * rperm(HD_x)

    # HGPHD_x
    fht(HGPHD_x)
    HGPHD_x /= float(tree['new_ncols'])

    return HGPHD_x
# -- end function

def search_ff_kdtree(tree, q) :
    n = tree['tree']
    qprojs = HGPHD_x(tree, q)
    new_ncols = tree['new_ncols']
    assert len(qprojs) == new_ncols
    while not n.leaf :
        if qprojs[n.level % new_ncols] < n.val :
            n = n.lchild
        else :
            n = n.rchild
    return n.pidxs
# -- end function
