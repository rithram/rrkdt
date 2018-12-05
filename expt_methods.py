from rptree import build_rptree, search_rptree
from rnd_rot_kdtree import build_rr_kdtree, search_rr_kdtree
from rnd_rot_kdtree import build_rconv_kdtree, search_rconv_kdtree
from sparse_rptree import build_sparse_rptree, search_sparse_rptree
from ff_kdtree import build_ff_kdtree, search_ff_kdtree

class HParams :
    leaf_size = None
    ntrees = None
# -- end class

def get_methods_for_expt(leaf_size, ntrees) :
    methods = []
    
    # plain rptree
    rp_hparam = HParams()
    rp_hparam.leaf_size = leaf_size
    rp_hparam.ntrees = ntrees
    rpt_method = {
        'name' : 'RPTree',
        'hparams' : rp_hparam,
        'indexer' : build_rptree,
        'locater' : search_rptree
    }
    methods.append(rpt_method)

    # sparse rptree with {Gaussian, Radamacher} distribution and p={1/3, 2/3}
    probs = [ (0.1, '1/10') , (1.0/3.0, '1/3') ]
    use_signs = [ (False, 'Ga'), (True, 'Ra') ]

    for u, s1 in use_signs :
        for p, s2 in probs :
            srp_hparam = HParams()
            srp_hparam.leaf_size = leaf_size
            srp_hparam.ntrees = ntrees
            srp_hparam.use_sign = u
            srp_hparam.col_prob = p
            srpt_method = {
                'name' : 'Sp' + s1 + ':RPT(' + s2 + ')',
                'hparams' : srp_hparam,
                'indexer' : build_sparse_rptree,
                'locater' : search_sparse_rptree
            }
            methods.append(srpt_method)
            
    # Randomly rotated data + k-d tree
    rrkd_hparam = HParams()
    rrkd_hparam.leaf_size = leaf_size
    rrkd_hparam.ntrees = ntrees
    rrkdt_method = {
        'name' : 'RR:KDTree',
        'hparams' : rrkd_hparam,
        'indexer' : build_rr_kdtree,
        'locater' : search_rr_kdtree
    }
    methods.append(rrkdt_method)

    # Randomly convolved data + k-d tre
    rckd_hparam = HParams()
    rckd_hparam.leaf_size = leaf_size
    rckd_hparam.ntrees = ntrees
    rckdt_method = {
        'name' : 'RC:KDTree',
        'hparams' : rckd_hparam,
        'indexer' : build_rconv_kdtree,
        'locater' : search_rconv_kdtree
    }
    methods.append(rckdt_method)

    # FastFood data + k-d tree
    ffkd_hparam = HParams()
    ffkd_hparam.leaf_size = leaf_size
    ffkd_hparam.ntrees = ntrees
    ffkdt_method = {
        'name' : 'FF:KDTree',
        'hparams' : ffkd_hparam,
        'indexer' : build_ff_kdtree,
        'locater' : search_ff_kdtree
    }
    methods.append(ffkdt_method)

    return methods
# -- end function
