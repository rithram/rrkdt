import numpy as np
from collections import deque

from rptree import rplog
from rptree import Node
from rptree import split_node

from ffht import fht
from mat_utils import fwht

def rperm(P_seed, x) :
    np.random.seed(P_seed)
    return np.random.permutation(x)
# -- end function

def HGPHD_x(D_by_sqrt_d, pad, P_seed, G, x) :
    # (1/d) * HDx
    HDx = np.concatenate([ x * D_by_sqrt_d, pad ])
    fht(HDx)
    # (1/d) * GPHDx
    HGPHDx = G * rperm(P_seed, HDx)
    # (1/d) * HGPHDx
    fht(HGPHDx)

    return HGPHDx
# -- end function

def H0GPH0D_x(D_by_sqrt_d, pad, P_seed, G, x) :
    # (1/d) * HDx
    HDx = np.concatenate([ x * D_by_sqrt_d, pad ])
    fwht(HDx)
    # (1/d) * GPHDx
    HGPHDx = G * rperm(P_seed, HDx)
    # (1/d) * HGPHDx
    fwht(HGPHDx)

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
    D = np.random.binomial(n=1, p=0.5, size=ncols).astype(float) * 2.0 - 1.0

    # Pad each point to have some power of 2 size
    lncols = np.log2(ncols)
    new_ncols = ncols if int(lncols) == lncols else np.power(2, int(lncols) + 1)
    logr('Padding %i features to %i with 0' % (ncols, new_ncols))
    pad_vec = np.zeros(new_ncols - ncols)

    # Cache the (1/d) operation inside the D sign vector
    D_by_sqrt_d = D / float(new_ncols)
    
    # Generate a random permutation matrix P
    P_seed = np.random.randint(9999)
    # Generate a random diagonal gaussian matrix G
    G = np.random.normal(size=new_ncols)

    HGPHD_S = np.array([
        HGPHD_x(D_by_sqrt_d, pad_vec, P_seed, G, p)
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
        'D_by_sqrt_d' : D_by_sqrt_d,
        'pad' : pad_vec,
        'P_seed' : P_seed,
        'G' : G
    }
# -- end function

def traverse_ff_kdtree(tree, log=False) :
    logr = lambda message : rplog(message, log)
    nodes = deque()
    nodes.append(tree['tree'])
    print('D:', tree['D'])
    print('G:', tree['G'])
    print('P_seed:', tree['P_seed'])
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

def search_ff_kdtree(tree, q, optimized=True) :
    n = tree['tree']
    new_ncols = tree['new_ncols']
    FFx = HGPHD_x if optimized else H0GPH0D_x
    qprojs = FFx(
        tree['D_by_sqrt_d'], tree['pad'], tree['P_seed'], tree['G'], q
    )
    while not n.leaf :
        if qprojs[n.level % new_ncols] < n.val :
            n = n.lchild
        else :
            n = n.rchild
    return n.pidxs
# -- end function
