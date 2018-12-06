import numpy as np
from collections import deque

from ffht import fht

from rptree import rplog
from rptree import Node
from rptree import split_node

def HD_x(D, pad, sqrt_new_ncols, x) :
    # [ Dx 0 ... 0 ]
    HDx = np.concatenate([ D * x, pad ])
    # H [ Dx 0 ... 0 ]
    fht(HDx)
    # (d^{-1/2}) * H [ Dx 0 ... 0 ]
    HDx /= sqrt_new_ncols

    return HDx
# -- end function


def compute_proj(q, col_idxs) :
    projs = [ 0.0, 0.0 ]
    for cidx, pidx in col_idxs :
        projs[pidx] += q[cidx]
    return (projs[1] - projs[0])
# -- end function

def build_sparse_rptree(S, hparams, log=False) :
    logr = lambda message : rplog(message, log)
    
    nrows, ncols = S.shape
    leaf_size = hparams.leaf_size
    logr(
        'Building sparse RP-tree on %i points in %i dims;\nmax. leaf size: %i'
        ', column choice Bernoulli probability %g, \nuse sign random variables %s'
        % (nrows, ncols, leaf_size, hparams.col_prob, str(hparams.use_sign))
    )

    # Generate a random diagonal sign matrix
    D = np.random.binomial(n=1, p=0.5, size=ncols) * 2 - 1

    # Pad each point to have some power of 2 size
    lncols = np.log2(ncols)
    new_ncols = ncols if int(lncols) == lncols else np.power(2, int(lncols) + 1)
    logr('Padding %i features to %i with 0' % (ncols, new_ncols))
    pad_vec = np.zeros(new_ncols - ncols)

    HD_S = np.array([
        HD_x(D, pad_vec, np.sqrt(new_ncols), p) for p in S
    ])
    
    a, b = HD_S.shape
    assert a == nrows and b == new_ncols

    logr('Densified data matrix has shape %s previously %s'
         % (repr(HD_S.shape), repr(S.shape))
    )

    nodes = deque()
    nidx = 0
    root = Node(0, 0)
    nodes.append((root, root.level, range(nrows)))

    level_col_idx = []
    level_rnd_vals = []
    all_projs_level = None
    
    while len(nodes) > 0 :
        n, l, idxs = nodes.popleft()
        indent = str('|-') * l;
        logr(
            '%sLevel %i, node %i, %i points ....' 
            % (indent, l, n.idx, len(idxs))
        )
        if l == len(level_col_idx) :
            # this level has no projections yet
            # 1. Choose the column indices for this level
            # 2. Generate a random values for these indices
            #  a. Choose between {-1, +1} or random normal
            # 3. Project all points in the table along this random vector
            tries = 0
            cidxs = []
            while tries < 10 and len(cidxs) == 0 :
                tries += 1
                cidx_indicators = np.random.binomial(n=1, p=hparams.col_prob, size=ncols)
                # FIXME:
                cidxs = [ idx for idx, b in enumerate(cidx_indicators) if b == 1 ]

            if len(cidxs) == 0 :
                raise Exception('No column got selected for projection (after 10 tries)')

            if hparams.use_sign :
                sign_vals = np.random.binomial(n=1, p=0.5, size=len(cidxs))
                all_idxs = zip(cidxs, sign_vals.astype(int))
                poss, negs = [], []
                for sv, cidx in zip(sign_vals, cidxs) :
                    if sv == 0 :
                        negs.append(cidx)
                    else :
                        poss.append(cidx)
                all_projs_level = (
                    np.sum(HD_S[:, poss], axis=1) - np.sum(HD_S[:, negs], axis=1)
                )
                level_col_idx.append(all_idxs)
            else :
                hp = np.random.normal(size=len(cidxs))
                all_projs_level = np.dot(HD_S[:, cidxs], hp)
                level_rnd_vals.append(hp)
                level_col_idx.append(cidxs)

        nidx = split_node(
            hparams.leaf_size,
            all_projs_level,
            idxs,
            indent,
            n,
            nidx,
            l + 1,
            nodes,
            logr
        )

    common_dict = {
        'tree' : root,
        'pad' : pad_vec,
        'D' : D,
        'ncols' : ncols,
        'new_ncols' : new_ncols,
        'sq_new_ncols' : np.sqrt(float(new_ncols)),
        'level_col_idx' : level_col_idx
    }

    if hparams.use_sign :
        assert len(level_rnd_vals) == 0
    else :
        common_dict['level_rnd_vals'] = level_rnd_vals

    return common_dict
# -- end function

def traverse_sparse_rptree(tree, log=False) :
    logr = lambda message : rplog(message, log)
    nodes = deque()
    nodes.append(tree['tree'])
    D = np.transpose(tree['D'])
    print('Diagonal sign matrix:', D)

    print('New column length:', tree['new_ncols'])

    level_col_idx = tree['level_col_idx']
    level_rnd_vals = tree['level_rnd_vals'] if 'level_rnd_vals' in tree else []
    use_sign = len(level_rnd_vals) == 0
    
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
            if use_sign :
                pidxs, nidxs = level_col_idx[n.level]
                ms = 'Col idxs: (' + str(pidxs) + ' - ' + str(nidxs) + '), val:' + str(n.val)
            else :
                ms = 'Col idxs:' + str(level_col_idx[n.level]) \
                     + ', hp:' + str(level_rnd_vals[n.level]) + ', val:' + str(n.val)
        logr(
            '%sL %i: leaf?%i, id:%i --> %s' 
            % (indent, l, n.leaf, n.idx, ms)
        )
# -- end function

def search_tree(root, qprojs) :
    n = root
    while not n.leaf :
        if qprojs[n.level] < n.val :
            n = n.lchild
        else :
            n = n.rchild
    return n.pidxs
# -- end function

def search_signed_sparse_rptree(tree, q) :
    HDq = HD_x(tree['D'], tree['pad'], tree['sq_new_ncols'] , q)
    qprojs = [
        compute_proj(HDq, col_idxs)
        for col_idxs in tree['level_col_idx']
    ]
    return search_tree(tree['tree'], qprojs)
# -- end function

def search_normal_sparse_rptree(tree, q) :
    HDq = HD_x(tree['D'], tree['pad'], tree['sq_new_ncols'] , q)
    qprojs = [
        np.dot(HDq[cidxs], hp)
        for cidxs, hp in zip(tree['level_col_idx'], tree['level_rnd_vals'])
    ]
    return search_tree(tree['tree'], qprojs)
# -- end function

def search_sparse_rptree(tree, q) :
    if 'level_rnd_vals' in tree :
        return search_normal_sparse_rptree(tree, q)
    else :
        return search_signed_sparse_rptree(tree, q)
# -- end function
