import numpy as np
from collections import deque

from rptree import rplog
from rptree import Node
from rptree import split_node

from ffht import fht

def rperm(P, x, length) :
    Px = np.zeros(length)
    for i in range(length) :
        Px[P[i]] = x[i]
    return Px
# -- end function

def HGPHD_x(D, pad, P, G, new_ncols, x) :
    # HDx
    HDx = np.concatenate([ x * D, pad ])
    fht(HDx)

    # GPHDx
    HGPHDx = G * rperm(P, HDx, new_ncols)

    # HGPHDx
    fht(HGPHDx)
    HGPHDx /= float(new_ncols)

    return HGPHDx
# -- end function

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
    D = np.random.binomial(n=1, p=0.5, size=ncols) * 2 - 1

    # Pad each point to have some power of 2 size
    lncols = np.log2(ncols)
    new_ncols = ncols if int(lncols) == lncols else np.power(2, int(lncols) + 1)
    logr('Padding %i features to %i with 0' % (ncols, new_ncols))
    pad_vec = np.zeros(new_ncols - ncols)

    # Generate a random permutation matrix P
    P = np.random.permutation(range(new_ncols))
    # Generate a random diagonal gaussian matrix G
    G = np.random.normal(size=new_ncols)

    HGPHD_S = np.array([
        HGPHD_x(D, pad_vec, P, G, new_ncols, p)
        for p in S
    ])
    
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
        'ncols' : ncols,
        'new_ncols' : new_ncols,
        'D' : D,
        'pad' : pad_vec,
        'P' : P,
        'G' : G
    }
# -- end function

def traverse_ff_kdtree(tree, log=False) :
    logr = lambda message : rplog(message, log)
    nodes = deque()
    nodes.append(tree['tree'])
    print('D:', tree['D'])
    print('G:', tree['G'])
    print('P:', tree['P'])
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

def search_ff_kdtree(tree, q) :
    n = tree['tree']
    new_ncols = tree['new_ncols']
    qprojs = HGPHD_x(
        tree['D'], tree['pad'], tree['P'], tree['G'], new_ncols, q
    )
    while not n.leaf :
        if qprojs[n.level % new_ncols] < n.val :
            n = n.lchild
        else :
            n = n.rchild
    return n.pidxs
# -- end function
