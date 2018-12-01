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
