import numpy as np
from collections import deque

def rplog(message, log) :
    if log :
        print(message)

class Node :
    def __init__(self, idx, level) :
        self.leaf = True
        self.idx = idx
        self.level = level
        
    def split_node(self, lc, rc, val) :
        self.leaf = False
        self.lchild = lc
        self.rchild = rc
        self.val = val
    
    def set_leaf(self, pidxs) :
        assert self.leaf
        self.pidxs = pidxs

def split_node(
    min_leaf_size,
    all_projections,
    idxs,
    indent,
    current_node,
    node_idx_offset,
    level_idx,
    nodes_list,
    logr
) :

    projs = [ all_projections[i] for i in idxs ]
    # shifting the median to fix numerical issues
    val = np.median(projs) * (1.0 + 1e-6)

    lidx = []
    ridx = []
    for i, p in zip(idxs, projs) :
        if p < val :
            lidx.append(i)
        else :
            ridx.append(i)

    if len(lidx) < min_leaf_size or len(ridx) < min_leaf_size :
        current_node.set_leaf(idxs)
        logr('%s  --> Setting node %i as leaf' % (indent, current_node.idx))
        return node_idx_offset

            
    #if len(lidx) == 0 or len(ridx) == 0 :
    #    print('Projs:', projs)
    #    print('val', val)
    #    print('median', np.median(projs))
    #    print('All projs:', all_projections)
    #    exit
    
    lc = Node(node_idx_offset + 1, level_idx)
    rc = Node(node_idx_offset + 2, level_idx)
    logr(
        '%s  --> Splitting node %i into %i-%i' 
        % (indent, current_node.idx, len(lidx), len(ridx))
    )

    current_node.split_node(lc, rc, val)
    nodes_list.append((lc, lc.level, lidx))
    nodes_list.append((rc, rc.level, ridx))

    return node_idx_offset + 2
# -- end function

def build_rptree(S, hparams, log=False) :
    logr = lambda message : rplog(message, log)

    nrows, ncols = S.shape
    leaf_size = hparams.leaf_size
    logr(
        'Building RP-tree on %i points in %i dims; max. leaf size: %i' 
        % (nrows, ncols, leaf_size)
    )
    nodes = deque()
    nidx = 0
    root = Node(0, 0)
    nodes.append((root, root.level, range(nrows)))
    hp_list = []
    all_projs_level = None
    
    while len(nodes) > 0 :
        n, l, idxs = nodes.popleft()
        indent = str('|-') * l;
        logr(
            '%sLevel %i, node %i, %i points ....' 
            % (indent, l, n.idx, len(idxs))
        )
        if l == len(hp_list) :
            # this level has no projections yet
            # 1. Generate a random vector
            # 2. Project all points in the table along this random vector
            hp = np.random.normal(size=ncols)
            all_projs_level = S @ hp
            hp_list.append(hp)

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

    return { 'tree' : root, 'projs' : np.transpose(np.array(hp_list)) }
# -- end function

def traverse_rptree(tree, log=False) :
    logr = lambda message : rplog(message, log)
    nodes = deque()
    nodes.append(tree['tree'])
    hps = np.transpose(tree['projs'])

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
            ms = 'hp:' + str(hps[n.level]) + ', val:' + str(n.val)
        logr(
            '%sL %i: leaf?%i, id:%i --> %s' 
            % (indent, l, n.leaf, n.idx, ms)
        )
# -- end function

def search_rptree(tree, q) :
    n = tree['tree']
    qprojs = np.dot(q, tree['projs'])
    while not n.leaf :
        if qprojs[n.level] < n.val :
            n = n.lchild
        else :
            n = n.rchild
    return n.pidxs
# -- end function

def search_rptree0(tree, q) :
    n = tree['tree']
    #qprojs = [ np.dot(q, proj) for proj in tree['projs'] ]
    while not n.leaf :
        qproj = np.dot(q, tree['projs'][:, n.level])
        if qproj < n.val :
            n = n.lchild
        else :
            n = n.rchild
    return n.pidxs
# -- end function

def mydot(q, w) :
    proj = 0.
    for x, y in zip(q, w) :
        proj += (x * y)
    return proj
# -- end function

def search_rptree00(tree, q) :
    n = tree['tree']
    while not n.leaf :
        w = tree['projs'][:, n.level]
        qproj = mydot(q, w)
        if qproj < n.val :
            n = n.lchild
        else :
            n = n.rchild
    return n.pidxs
# -- end function
